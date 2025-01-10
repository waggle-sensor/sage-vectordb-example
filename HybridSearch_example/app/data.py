'''This file contains code that adds data to weaviate using sage_data_client.
These images will be the ones with which the hybrid search will compare
the text query given by the user.'''

import weaviate
import uuid
import os
import sage_data_client
import requests
import cv2
from setup import setup_client
import shutil
from transformers import AutoProcessor, AutoModelForCausalLM
from model import generate_caption

MODEL_PATH = os.environ.get("MODEL_PATH")

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

    # Initiate Model and Processor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True)

    # Retrieve the Sage configuration
    sage_username = username
    sage_token = token

    #auth header
    auth = (sage_username, sage_token)

    client = setup_client(client)

    try:
        _locals = locals()
        exec(query,{'sage_data_client': sage_data_client}, _locals) 
        df = _locals['df']
    except Exception as e:
        print("Error:", e)

    # Create a directory to save images if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in df.index:
        url = df.value[i]
        timestamp = df.timestamp[i]
        try:
            response = requests.get(url, auth=auth)
            response.raise_for_status()  # Raise an error for bad responses
            image_data = response.content

            img_filename = f"image_{i}.jpg"
            full_path = os.path.join(save_dir, img_filename)
            with open(full_path, 'wb') as f:
                f.write(image_data)

            # Encode the image using the temporary file path
            encoded_image = weaviate.util.image_encoder_b64(full_path)
            
            # Load the image
            image = cv2.imread(full_path)

            #generate caption
            caption = generate_caption(model, processor, image)

            #get collection
            collection = client.collections.get("HybridSearchExample")

            #create data object in Weaviate
            data_properties = {
                "filename": img_filename,
                "image": encoded_image,
                "timestamp": timestamp.strftime('%y-%m-%d %H:%M Z'),
                "link": url,
                "caption": caption
            }

            collection.data.insert(properties=data_properties,uuid=generate_uuid('HybridSearchExample', str(i)))

        except requests.exceptions.HTTPError as e:
            print('Image skipped ' + url)

    print("Images and Captions added")

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