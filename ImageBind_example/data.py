'''This file contains code that adds data to weaviate using sage_data_client.
These images will be the ones with which the module multi2vec-bind will compare
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
import configparser
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from setup import setup_client

# Parse the configuration file
config = configparser.ConfigParser()
config.read('../config.ini')

# Retrieve the Sage configuration
sage_username = config['Sage']['username']
sage_token = config['Sage']['token']

#auth header
auth = (sage_username, sage_token)

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

client = setup_client()

df = sage_data_client.query(
    start="-24h",
    #end="2023-02-22T23:00:00.000Z",
    filter={
        "plugin": "registry.sagecontinuum.org/theone/imagesampler.*",
        "vsn": "W088"
        #"job": "imagesampler-top"
    }
).sort_values('timestamp')

# Create a directory to save images if it doesn't exist
save_dir = "static/Images"
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