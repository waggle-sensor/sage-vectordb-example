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

MANIFEST_API = os.environ.get("MANIFEST_API")

def watch(start=None, filter=None):
    """
    Watches for incoming data and yields dataframes as new data is available.
    """
    if start is None:
        start = pd.Timestamp.utcnow()

    while True:
        df = sage_data_client.query(
            start=start,
            filter=filter
        )

        if len(df) > 0:
            start = df.timestamp.max()
            yield df

        time.sleep(3.0)

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
    
    logger.debug(f"[DATA] Processing image: {vsn}, {timestamp}, {url}")
    
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
        project = manifest.get('project', '')
        address = manifest.get('address', '')
        lat = manifest.get('gps_lat', '')
        lon = manifest.get('gps_lon', '')

        # Get live lat & lon
        loc_df = sage_data_client.query(start="-5m", filter={"vsn": vsn.upper(), "name": "sys.gps.lat|sys.gps.lon"}, tail=1)
        if not loc_df.empty:
            lat = loc_df[loc_df['name'] == 'sys.gps.lat']['value'].values[0]
            lon = loc_df[loc_df['name'] == 'sys.gps.lon']['value'].values[0]

        # Generate caption
        caption = gemma3_run_model(triton_client, image)

        # Generate clip embedding
        clip_embedding = get_clip_embeddings(triton_client, caption, image)

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
            "location": GeoCoordinate(latitude=float(lat), longitude=float(lon)),
        }

        collection.data.insert(
            properties=data_properties,
            vector={"clip": clip_embedding}
        )
        
        logger.debug(f'[DATA] Image added: {url}')
        return {"status": "success", "url": url, "vsn": vsn}

    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTPError for URL {url}: {e}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed for URL {url}: {e}")
    except Exception as e:
        raise Exception(f"Error processing URL {url}: {e}")

# def continual_load(username, token, weaviate_client, triton_client):
#     '''
#     Continously Load data to weaviate
#     '''

#     # Retrieve the Sage configuration
#     sage_username = username
#     sage_token = token

#     # Auth header for Sage
#     auth = (sage_username, sage_token)

#     # Setup filter to query specific data
#     filter = {
#         "plugin": "registry.sagecontinuum.org/yonghokim/imagesampler.*"
#     }

#     # Watch for data in real-time
#     for df in watch(start=None, filter=filter):
#         vsns = df['meta.vsn'].unique()
#         end_time = df.timestamp.max()
#         start_time = df.timestamp.min()

#         logging.debug(f'Start processing images for the following nodes: {vsns}')
#         logging.debug(f'Start time: {start_time}, End time: {end_time}')
#         logging.debug('')

#         for i in df.index:
#             url = df.value[i]
#             timestamp = df.timestamp[i]
#             vsn = df["meta.vsn"][i]
#             filename = df["meta.filename"][i]
#             camera = df["meta.camera"][i]
#             host = df["meta.host"][i]
#             job = df["meta.job"][i]
#             node = df["meta.node"][i]
#             plugin = df["meta.plugin"][i]
#             task = df["meta.task"][i]
#             zone = df["meta.zone"][i]

#             logging.debug(f"Image info: {vsn}, {timestamp}, {url}")
#             try:
#                 # Get the image data
#                 response = requests.get(url, auth=auth)
#                 try:
#                     response.raise_for_status() # Raise error for bad responses
#                 except requests.HTTPError as e:
#                     logging.debug(f"Request failed with status {response.status_code}: {e}")
#                     continue
#                 image_data = response.content

#                 # Check if the response contains valid image data
#                 if not image_data:
#                     logging.debug(f"Image skipped, empty content received for URL: {url}")
#                     continue

#                 # Wrap the BytesIO stream in BufferedReader
#                 image_stream = BytesIO(image_data)
#                 buffered_stream = BufferedReader(image_stream)

#                 #Reset the pointer to the beginning
#                 image_stream.seek(0)  

#                 # Encode the image
#                 encoded_image = weaviate.util.image_encoder_b64(buffered_stream)

#                 # Reset the pointer to the beginning, to be used again
#                 image_stream.seek(0)
#                 try:
#                     image = Image.open(image_stream).convert("RGB")
#                 except (OSError, IOError) as e:
#                     logging.debug(f"Image open failed: {e}")
#                     continue

#                 # Get the manifest
#                 response = requests.get(urljoin(MANIFEST_API, vsn.upper()))
#                 response.raise_for_status()  # Raise error for bad responses
#                 manifest = response.json()

#                 # Extract fields from manifest
#                 project = manifest.get('project', '')
#                 address = manifest.get('address', '')
#                 lat = manifest.get('gps_lat', '')
#                 lon = manifest.get('gps_lon', '')

#                 # Get live lat & lon
#                 loc_df = sage_data_client.query(start="-5m", filter={"vsn": vsn.upper(), "name": "sys.gps.lat|sys.gps.lon"}, tail=1)
#                 if not loc_df.empty:
#                     # Extract
#                     lat = loc_df[loc_df['name'] == 'sys.gps.lat']['value'].values[0]
#                     lon = loc_df[loc_df['name'] == 'sys.gps.lon']['value'].values[0]

#                 # Generate caption
#                 caption = gemma3_run_model(triton_client, image)

#                 # Generate clip embedding
#                 clip_embedding = get_clip_embeddings(triton_client, caption, image)

#                 # Get Weaviate collection
#                 collection = weaviate_client.collections.get("HybridSearchExample")

#                 # Prepare data for insertion into Weaviate
#                 data_properties = {
#                     "filename": filename,
#                     "image": encoded_image,
#                     "timestamp": timestamp.strftime('%y-%m-%d %H:%M Z'),
#                     "link": url,
#                     "caption": caption,
#                     "camera": camera,
#                     "host": host,
#                     "job": job,
#                     "node": node,
#                     "plugin": plugin,
#                     "task": task,
#                     "vsn": vsn,
#                     "zone": zone,
#                     "project": project,
#                     "address": address,
#                     "location": GeoCoordinate(latitude=float(lat), longitude=float(lon)),
#                 }

#                 collection.data.insert(
#                     properties=data_properties,
#                     vector={"clip": clip_embedding}
#                     )
#                 logging.debug(f'Image added: {url}')

#             except requests.exceptions.HTTPError as e:
#                 logging.debug(f"Image skipped, HTTPError for URL {url}: {e}")
#             except requests.exceptions.RequestException as e:
#                 logging.debug(f"Image skipped, request failed for URL {url}: {e}")
#             except Exception as e:
#                 logging.debug(f"Image skipped, an error occurred for URL {url}: {e}")

#     logging.debug("Images and Captions added to Weaviate")
