'''This file implements functions that fetch results from weaviate for the query 
entered by user. There are two functions, testImage and testText for image query and text query
respectively.'''

import pickle
import weaviate
import uuid
import datetime
import base64, json, os

client = weaviate.Client("http://weaviate:8080")
print("Client created (This is test.py)")

def testText(nearText):
    # I am fetching top 3 results for the user, we can change this by making small 
    # altercations in this function and in upload.html file
    # You can also analyse the result in a better way by taking a look at res.
    # Try printing res in the terminal and see what all contents it has.
    # adding a threshold: https://weaviate.io/developers/weaviate/search/similarity#set-a-similarity-threshold
    res = client.query.get("ClipExample", ["text", "_additional {certainty} "]).with_near_text(nearText).do()
    return {
        "objects": ((res['data']['Get']['ClipExample'][0]['text']),(res['data']['Get']['ClipExample'][1]['text']),(res['data']['Get']['ClipExample'][2]['text'])),
        "scores": (res['data']['Get']['ClipExample'][0]['_additional'],res['data']['Get']['ClipExample'][1]['_additional'],res['data']['Get']['ClipExample'][2]['_additional'])
    }


def testImage(nearImage):
    # I am fetching top 3 results for the user, we can change this by making small 
    # altercations in this function and in upload.html file
    # # adding a threshold: https://weaviate.io/developers/weaviate/search/similarity#set-a-similarity-threshold
    imres = client.query.get("ClipExample", ["text", "_additional {certainty} "]).with_near_image(nearImage).do()
    return {
        "objects": ((imres['data']['Get']['ClipExample'][0]['text']),(imres['data']['Get']['ClipExample'][1]['text']),(imres['data']['Get']['ClipExample'][2]['text'])),
        "scores": (imres['data']['Get']['ClipExample'][0]['_additional'],imres['data']['Get']['ClipExample'][1]['_additional'],imres['data']['Get']['ClipExample'][2]['_additional'])
    }