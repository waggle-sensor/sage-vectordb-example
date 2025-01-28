'''This file contains code that adds data to weaviate using sage_data_client.
These images will be the ones with which the hybrid search will compare
the text query given by the user.'''
#NOTE: This will be deployed in our cloud under k8s namespace beehive-sage
#   most likely integrated with sage-data-loader. Keep in mind, I will have to
#   somehow make the data loader not wait on creating an object in weaviate
#   because this takes longer.

import weaviate
import uuid
import os
import sage_data_client
import requests
import logging
from setup import setup_collection
import shutil
from transformers import AutoProcessor, AutoModelForCausalLM
from model import generate_caption, triton_gen_caption
from urllib.parse import urljoin
import tritonclient.grpc as TritonClient

MODEL_PATH = os.environ.get("MODEL_PATH")
MANIFEST_API = os.environ.get("MANIFEST_API")

def generate_uuid(class_name: str, identifier: str,
                test: str = 'teststrong') -> str:
    """ Generate a uuid based on an identifier
    :param identifier: characters used to generate the uuid
    :type identifier: str, required
    :param class_name: classname of the object to create a uuid for
    :type class_name: str, required
    """
    test = 'overwritten'
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, class_name + identifier))

def load_data(username, token, query, client, save_dir="static/Images"):
    '''
    Load data to weaviate and objects to save_dir
    '''

    # Initiate Local Model and Processor
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_PATH,
    #     local_files_only=True,
    #     trust_remote_code=True)
    # processor = AutoProcessor.from_pretrained(
    #     MODEL_PATH,
    #     local_files_only=True,
    #     trust_remote_code=True)

    #init triton client
    triton_client = TritonClient.InferenceServerClient(url="florence2:8001")

    # Retrieve the Sage configuration
    sage_username = username
    sage_token = token

    #auth header
    auth = (sage_username, sage_token)

    #set up collection that will hold vectors and metadata
    setup_collection(client)

    try:
        _locals = locals()
        exec(query,{'sage_data_client': sage_data_client}, _locals) 
        df = _locals['df']
    except Exception as e:
        logging.error("Error:", e)

    # Find all columns starting with 'meta.'
    meta_columns = [col for col in df.columns if col.startswith('meta.')]

    # Concatenate all meta columns into a single string
    df['meta_combined'] = df[meta_columns].apply(lambda row: ' '.join(row.astype(str)), axis=1)

    # Create a directory to save images if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in df.index:
        url = df.value[i]
        timestamp = df.timestamp[i]
        meta = df.meta_combined[i]
        vsn= df["meta.vsn"][i]
        try:
            response = requests.get(url, auth=auth)
            response.raise_for_status()  # Raise an error for bad responses
            image_data = response.content

            # Check if the response contains valid image data
            if not image_data:
                logging.debug(f"Image skipped, empty content received for URL: {url}")
                continue  # Skip to the next iteration if the image is empty

            img_filename = f"image_{i}.jpg"
            full_path = os.path.join(save_dir, img_filename)
            with open(full_path, 'wb') as f:
                f.write(image_data)

            # Encode the image using the temporary file path
            encoded_image = weaviate.util.image_encoder_b64(full_path)

            #Get Manifest
            response = requests.get(urljoin(MANIFEST_API,vsn.upper()))
            response.raise_for_status()  # Raise an error for bad responses
            manifest = response.json() 

            # Extract only 'project' and 'address' from the manifest response
            project = manifest.get('project', '')
            address = manifest.get('address', '')

            # Combine 'project' and 'address' into the metadata
            meta_combined = f"{meta} {project} {address}"

            # Add the combined metadata to the DataFrame
            df.at[i, 'meta_combined'] = meta_combined

            #Generate caption
            # caption = generate_caption(model, processor, full_path)
            caption = triton_gen_caption(triton_client,full_path)

            #get collection
            collection = client.collections.get("HybridSearchExample")

            #create data object in Weaviate
            data_properties = {
                "filename": img_filename,
                "image": encoded_image,
                "timestamp": timestamp.strftime('%y-%m-%d %H:%M Z'),
                "link": url,
                "caption": caption,
                "meta": meta
            }

            collection.data.insert(properties=data_properties,uuid=generate_uuid('HybridSearchExample', str(i)))
            logging.debug('Image added ' + url)

        except requests.exceptions.HTTPError as e:
            logging.debug(f"Image skipped, HTTPError for URL {url}: {e}")

        except requests.exceptions.RequestException as e:
            logging.debug(f"Image skipped, request failed for URL {url}: {e}")

        except Exception as e:
            logging.debug(f"Image skipped, an error occurred for URL {url}: {e}")

    logging.debug("Images and Captions added to Weaviate")

def clear_data(dir="static/Images"):
    '''
    Check if the directory exists and remove it
    '''
    if os.path.exists(dir):
        shutil.rmtree(dir)

def check_data(dir="static/Images"):
    """
    Check if there are files dir
    """
    if os.path.exists(dir):
        if os.listdir(dir):
            return "Images Loaded"  # Files exist in the directory
        else:
            return "Empty"  # Directory is empty
    else:
        return "Empty"  # Directory does not exist