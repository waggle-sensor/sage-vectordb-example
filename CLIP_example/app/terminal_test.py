'''This file is to check the code by running some test in terminal.
I have named the images such that it is easy while testing in terminal
without a frontend.'''

import pickle
import weaviate
import uuid
import datetime
import base64, json, os
from test import testImage,testText
from setup import setup_client

client = setup_client

# You can add more text input tests here
concepts = [
    'A red car',
    'cloudy day',
    'many people',
    'nuves', #spanish for cloud
    'stars in the sky',
    'dogs',
    'forest fire',
    'empty crosswalk'
]
print("==========================================")
for con in concepts:
    print(f"Input text --> {con} --> Result -->",testText({"concepts":[con]}))
print("==========================================")


# To add more image input texts, add test images to the static/Test folder.
print("==========================================")
test_images = os.listdir("static/Test/")
for img in test_images:
    print(f"Input Image --> {img} --> Result -->",testImage({"image":f"static/Test/{img}"}))
print("==========================================")