'''This file contains code that adds data to weaviate using sage_data_client.
These images will be the ones with which the module multi2-vec-clip will compare
the image or text query given by the user.'''

import pickle
import weaviate
import uuid
import datetime
import base64, json, os
import sage_data_client
import requests
import io
import tempfile
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from setup import setup_client
import shutil

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

            # Open the image using PIL
            img = Image.open(full_path)

            # Define a font for the text
            font_properties = font_manager.FontProperties(family='sans-serif', weight='bold')
            font_file = font_manager.findfont(font_properties)
            font = ImageFont.truetype(font_file, 60)

            # Draw text on the image
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), timestamp.strftime('%y-%m-%d %H:%M Z'), fill='white', font=font)

            # Save the modified image
            img.save(full_path)

            # Encode the image using the temporary file path
            encoded_image = weaviate.util.image_encoder_b64(full_path)

            # Create data object in Weaviate
            data_properties = {
                "image": encoded_image,
                "text": img_filename
            }

            client.data_object.create(data_properties, "ClipExample", generate_uuid('ClipExample', str(i)))
        except requests.exceptions.HTTPError as e:
            print('Image skipped ' + url)

    print("Images added")

    # You can try uncommenting the below code to add text as well
    # After adding the texts, these texts can also be fetched as results if their
    # embeddings are similar to the embedding of the query. Currently the frontend is
    # designed so as to accommodate these as well. 

    # Adding texts
    # texts = [
    #     'A dense forest',
    #     'A beautiful beach',
    #     'people playing games',
    #     'Students syudying in class',
    #     'a beautiful painting',
    #     'place with scenic beauty',
    #     'confident woman',
    #     'cute little creature',
    #     'players playing badminton'
    # ]
    # for txt in texts:
    #     data_properties = {
    #         "text":txt
    #     }
    #     client.data_object.create(data_properties, "ClipExample", generate_uuid('ClipExample',txt))
    # print("Texts added")

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
