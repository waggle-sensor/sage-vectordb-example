'''This file contains code that adds data to weaviate using sage_data_client.
These images will be the ones with which the hybrid search will compare
the text query given by the user.'''

import weaviate
import os
import pandas as pd
import time
import sage_data_client
import requests
import logging
from PIL import Image
from io import BytesIO, BufferedReader
from inference import gemma3_run_model, get_clip_embeddings, qwen2_5_run_model
from urllib.parse import urljoin
from weaviate.classes.data import GeoCoordinate
from metrics import metrics
import numpy as np

MANIFEST_API = os.environ.get("MANIFEST_API", "https://auth.sagecontinuum.org/manifests/")

def watch(start=None, filter=None, logger=logging.getLogger(__name__)):
    """
    Watches for incoming data and yields dataframes as new data is available.
    Uses adaptive polling to minimize traffic:
    - Faster polling when data is found (burst detection)
    - Slower polling when no data (reduce idle traffic)
    """
    if start is None:
        start = pd.Timestamp.utcnow()
    
    # Configurable intervals (in seconds)
    min_interval = 15.0      # When data found
    max_interval = 120.0     # When no data
    current_interval = min_interval
    
    while True:
        try:
            df = sage_data_client.query(
                    start=start,
                    filter=filter
                )
            metrics.update_component_health('sage', True)
        except Exception as e:
            metrics.update_component_health('sage', False)
            logger.error(f"[PROCESSING] Error querying Sage data: {e}")
            current_interval = min(current_interval * 1.5, max_interval)
            logger.debug(f"[PROCESSING] Error querying Sage data, increasing interval to {current_interval:.1f}s")
            time.sleep(current_interval)
            continue
        
        if len(df) > 0:
            # Data found - use fast polling for burst detection
            start = df.timestamp.max()
            current_interval = min_interval
            logger.debug(f"[PROCESSING] Sage data found, resetting interval to {min_interval:.1f}s")
            yield df
        else:
            # No data - gradually increase interval to reduce idle traffic
            current_interval = min(current_interval * 1.5, max_interval)
            logger.debug(f"[PROCESSING] No data, increasing interval to {current_interval:.1f}s")

        time.sleep(current_interval)

def parse_deny_list(raw: str) -> set[str]:
    """
    Parse the deny list from the environment variable.
    Args:
        raw (str): The raw deny list string
        
    Returns:
        set[str]: The parsed deny list
    """
    return {x.strip().lower() for x in raw.split(",") if x.strip()}

def process_image(image_data, username, token, weaviate_client, triton_client, logger=logging.getLogger(__name__)):
    """
    Process a single image and add it to Weaviate.
    
    Args:
        image_data (dict): Dictionary containing image metadata
        username (str): SAGE username
        token (str): SAGE token
        weaviate_client: Weaviate client instance
        triton_client: Triton client instance
        
    Returns:
        dict: Processing result
    """
    url = image_data['url']
    timestamp = pd.Timestamp(image_data['timestamp'])
    vsn = image_data['vsn']
    filename = image_data['filename']
    camera = image_data['camera']
    host = image_data['host']
    job = image_data['job']
    node = image_data['node']
    plugin = image_data['plugin']
    task = image_data['task']
    zone = image_data['zone']
    
    # Auth header for Sage
    auth = (username, token)
    
    logger.debug(f"[PROCESSING] Processing image: {vsn}, {timestamp}, {url}")
    
    try:
        # Get the image data
        response = requests.get(url, auth=auth)
        response.raise_for_status()
        image_content = response.content

        # Check if the response contains valid image data
        if not image_content:
            raise ValueError(f"Empty content received for URL: {url}")

        # Wrap the BytesIO stream in BufferedReader
        image_stream = BytesIO(image_content)
        buffered_stream = BufferedReader(image_stream)

        # Reset the pointer to the beginning
        image_stream.seek(0)  

        # Encode the image
        encoded_image = weaviate.util.image_encoder_b64(buffered_stream)

        # Reset the pointer to the beginning, to be used again
        image_stream.seek(0)
        image = Image.open(image_stream).convert("RGB")

        # Get the manifest
        response = requests.get(urljoin(MANIFEST_API, vsn.upper()))
        response.raise_for_status()
        manifest = response.json()

        # Extract fields from manifest
        project = manifest.get('project', 'unknown')
        address = manifest.get('address', 'unknown')
        lat = manifest.get('gps_lat', 0.0)
        lon = manifest.get('gps_lon', 0.0)

        # Get live lat & lon
        loc_df = sage_data_client.query(start="-5m", filter={"vsn": vsn.upper(), "name": "sys.gps.lat|sys.gps.lon"}, tail=1)
        if not loc_df.empty:
            lat = loc_df[loc_df['name'] == 'sys.gps.lat']['value'].values[0]
            lon = loc_df[loc_df['name'] == 'sys.gps.lon']['value'].values[0]

        # Generate caption
        start_time = time.perf_counter()
        try:
            caption = gemma3_run_model(triton_client, image)
            caption_duration = time.perf_counter() - start_time
            metrics.record_model_inference("gemma3", "caption", caption_duration, "success")
        except Exception as e:
            caption_duration = time.perf_counter() - start_time
            metrics.record_model_inference("gemma3", "caption", caption_duration, "failure")
            raise e

        # Generate clip embedding
        start_time = time.perf_counter()
        try:
            clip_embedding = get_clip_embeddings(triton_client, caption, image)
            embedding_duration = time.perf_counter() - start_time
            metrics.record_model_inference("clip", "embedding", embedding_duration, "success")
        except Exception as e:
            embedding_duration = time.perf_counter() - start_time
            metrics.record_model_inference("clip", "embedding", embedding_duration, "failure")
            raise e

        # Get Weaviate collection
        collection = weaviate_client.collections.get("HybridSearchExample")

        # Prepare data for insertion into Weaviate
        data_properties = {
            "filename": filename,
            "image": encoded_image,
            "timestamp": timestamp.strftime('%y-%m-%d %H:%M Z'),
            "link": url,
            "caption": caption,
            "camera": camera,
            "host": host,
            "job": job,
            "node": node,
            "plugin": plugin,
            "task": task,
            "vsn": vsn,
            "zone": zone,
            "project": project,
            "address": address,
            "location": GeoCoordinate(latitude=float(lat) if lat is not None else 0.0, longitude=float(lon) if lon is not None else 0.0),
        }

        # Check if the clip embedding is finite
        if not np.all(np.isfinite(clip_embedding)):
            logger.error(f"[PROCESSING] Non-finite values in clip embedding for {url}: {clip_embedding}")
            raise ValueError(f"Non-finite values in clip embedding for {url}")

        # Insert into Weaviate with metrics
        start_time = time.perf_counter()
        try:
            collection.data.insert(
                properties=data_properties,
                vector={"clip": clip_embedding}
            )
            insert_duration = time.perf_counter() - start_time
            metrics.record_weaviate_operation("insert", "success", insert_duration)
        except Exception as e:
            insert_duration = time.perf_counter() - start_time
            metrics.record_weaviate_operation("insert", "failure", insert_duration)
            raise e
        
        logger.debug(f'[PROCESSING] Image added: {url}')
        return {"status": "success", "url": url, "vsn": vsn}

    except requests.exceptions.HTTPError as e:
        raise e
    except requests.exceptions.RequestException as e:
        raise e
    except Exception as e:
        raise e
