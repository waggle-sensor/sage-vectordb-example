'''This file implements functions that fetch results from weaviate for the query 
entered by user. There are two functions, testImage and testText for image query and text query
respectively.'''

import pickle
import weaviate
import uuid
import datetime
import base64, json, os

client = weaviate.Client("http://localhost:8080")
print("Client created (This is test.py)")

def testText(nearText):
    # I am fetching top 3 results for the user, we can change this by making small 
    # altercations in this function and in upload.html file
    # You can also analyse the result in a better way by taking a look at res.
    # Try printing res in the terminal and see what all contents it has.
    res = client.query.get("BindExample", ["text", "_additional {certainty} "]).with_near_text(nearText).do()
    #print certainty for top 3 results
    print(res['data']['Get']['BindExample'][0]['_additional'])
    print(res['data']['Get']['BindExample'][1]['_additional'])
    print(res['data']['Get']['BindExample'][2]['_additional'])
    return (res['data']['Get']['BindExample'][0]['text']),(res['data']['Get']['BindExample'][1]['text']),(res['data']['Get']['BindExample'][2]['text'])


def testImage(nearImage):
    # I am fetching top 3 results for the user, we can change this by making small 
    # altercations in this function and in upload.html file
    imres = client.query.get("BindExample", ["text", "_additional {certainty} "]).with_near_image(nearImage).do()
    #print certainty for top 3 results
    print(imres['data']['Get']['BindExample'][0]['_additional'])
    print(imres['data']['Get']['BindExample'][1]['_additional'])
    print(imres['data']['Get']['BindExample'][2]['_additional'])
    return (imres['data']['Get']['BindExample'][0]['text']),(imres['data']['Get']['BindExample'][1]['text']),(imres['data']['Get']['BindExample'][2]['text'])