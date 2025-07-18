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
from model import gemma3_run_model, get_clip_embeddings, qwen2_5_run_model
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

def continual_load(username, token, weaviate_client, triton_client):
    '''
    Continously Load data to weaviate
    '''

    # Retrieve the Sage configuration
    sage_username = username
    sage_token = token

    # Auth header for Sage
    auth = (sage_username, sage_token)

    # Setup filter to query specific data
    filter = {
        "plugin": "registry.sagecontinuum.org/yonghokim/imagesampler.*"
    }

    # Watch for data in real-time
    for df in watch(start=None, filter=filter):

        for i in df.index:
            url = df.value[i]
            timestamp = df.timestamp[i]
            vsn = df["meta.vsn"][i]
            filename = df["meta.filename"][i]
            camera = df["meta.camera"][i]
            host = df["meta.host"][i]
            job = df["meta.job"][i]
            node = df["meta.node"][i]
            plugin = df["meta.plugin"][i]
            task = df["meta.task"][i]
            zone = df["meta.zone"][i]

            try:
                # Get the image data
                response = requests.get(url, auth=auth)
                response.raise_for_status()  # Raise error for bad responses
                image_data = response.content

                # Check if the response contains valid image data
                if not image_data:
                    logging.debug(f"Image skipped, empty content received for URL: {url}")
                    continue

                # Wrap the BytesIO stream in BufferedReader
                image_stream = BytesIO(image_data)
                buffered_stream = BufferedReader(image_stream)

                #Reset the pointer to the beginning
                image_stream.seek(0)  

                # Encode the image
                encoded_image = weaviate.util.image_encoder_b64(buffered_stream)

                # Reset the pointer to the beginning, to be used again
                image_stream.seek(0)  
                image = Image.open(image_stream).convert("RGB")

                # Get the manifest
                response = requests.get(urljoin(MANIFEST_API, vsn.upper()))
                response.raise_for_status()  # Raise error for bad responses
                manifest = response.json()

                # Extract fields from manifest
                project = manifest.get('project', '')
                address = manifest.get('address', '')
                lat = manifest.get('gps_lat', '')
                lon = manifest.get('gps_lon', '')

                # Get live lat & lon
                loc_df = sage_data_client.query(start="-5m", filter={"vsn": vsn.upper(), "name": "sys.gps.lat|sys.gps.lon"}, tail=1)
                if not loc_df.empty:
                    # Extract
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
                logging.debug(f'Image added: {url}')

            except requests.exceptions.HTTPError as e:
                logging.debug(f"Image skipped, HTTPError for URL {url}: {e}")
            except requests.exceptions.RequestException as e:
                logging.debug(f"Image skipped, request failed for URL {url}: {e}")
            except Exception as e:
                logging.debug(f"Image skipped, an error occurred for URL {url}: {e}")

    logging.debug("Images and Captions added to Weaviate")