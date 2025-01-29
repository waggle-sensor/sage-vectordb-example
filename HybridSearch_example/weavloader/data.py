'''This file contains code that adds data to weaviate using sage_data_client.
These images will be the ones with which the hybrid search will compare
the text query given by the user.'''

import weaviate
import uuid
import os
import pandas as pd
import time
import sage_data_client
import requests
import logging
from model import triton_gen_caption
from urllib.parse import urljoin

MANIFEST_API = os.environ.get("MANIFEST_API")

def generate_uuid(class_name: str, identifier: str) -> str:
    """ Generate a uuid based on an identifier
    :param identifier: characters used to generate the uuid
    :type identifier: str, required
    :param class_name: classname of the object to create a uuid for
    :type class_name: str, required
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, class_name + identifier))

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

def continual_load(username, token, weaviate_client, triton_client, save_dir="static/Images"):
    '''
    Continously Load data to weaviate and objects to save_dir
    '''
    # init image index
    INDEX = 0

    # Retrieve the Sage configuration
    sage_username = username
    sage_token = token

    # Auth header for Sage
    auth = (sage_username, sage_token)

    # Setup filter to query specific data
    filter = {
        "plugin": "registry.sagecontinuum.org/yonghokim/imagesampler:*.*.*"
    }

    # Watch for data in real-time
    for df in watch(start=None, filter=filter):
        # Find all columns starting with 'meta.'
        meta_columns = [col for col in df.columns if col.startswith('meta.')]

        # Concatenate all meta columns into a single string
        df['meta_combined'] = df[meta_columns].apply(lambda row: ' '.join(row.astype(str)), axis=1)

        # Create directory for saving images if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in df.index:
            url = df.value[i]
            timestamp = df.timestamp[i]
            meta = df.meta_combined[i]
            vsn = df["meta.vsn"][i]

            try:
                # Get the image data
                response = requests.get(url, auth=auth)
                response.raise_for_status()  # Raise error for bad responses
                image_data = response.content

                # Check if the response contains valid image data
                if not image_data:
                    logging.debug(f"Image skipped, empty content received for URL: {url}")
                    continue

                # Save the image to the specified directory
                img_filename = f"image_{INDEX}.jpg"
                full_path = os.path.join(save_dir, img_filename)
                with open(full_path, 'wb') as f:
                    f.write(image_data)

                # Encode the image
                encoded_image = weaviate.util.image_encoder_b64(full_path)

                # Get the manifest
                response = requests.get(urljoin(MANIFEST_API, vsn.upper()))
                response.raise_for_status()  # Raise error for bad responses
                manifest = response.json()

                # Extract 'project' and 'address' from manifest
                project = manifest.get('project', '')
                address = manifest.get('address', '')

                # Combine the 'meta' fields
                meta = f"{meta} {project} {address}"

                # Generate caption
                caption = triton_gen_caption(triton_client, full_path)

                # Get Weaviate collection
                collection = weaviate_client.collections.get("HybridSearchExample")

                # Prepare data for insertion into Weaviate
                data_properties = {
                    "filename": img_filename,
                    "image": encoded_image,
                    "timestamp": timestamp.strftime('%y-%m-%d %H:%M Z'),
                    "link": url,
                    "caption": caption,
                    "meta": meta
                }

                collection.data.insert(properties=data_properties, uuid=generate_uuid('HybridSearchExample', str(INDEX)))
                logging.debug(f'Image added: {url}')
                INDEX += 1

            except requests.exceptions.HTTPError as e:
                logging.debug(f"Image skipped, HTTPError for URL {url}: {e}")
            except requests.exceptions.RequestException as e:
                logging.debug(f"Image skipped, request failed for URL {url}: {e}")
            except Exception as e:
                logging.debug(f"Image skipped, an error occurred for URL {url}: {e}")

    logging.debug("Images and Captions added to Weaviate")