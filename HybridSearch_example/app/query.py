'''This file implements functions that fetch results from weaviate for the query 
entered by user.'''
#NOTE: This will be deployed in our cloud under k8s namespace beehive-sage 
#   most likely integrated with beehive-data-api. If we want to allow our 
#   python client to use weaviate queries, we will also have to update our
#   sage-data-client python lib to include these new queries.

import HyperParameters as hp
from weaviate.classes.query import MetadataQuery, Move, HybridVector, Rerank
import logging
import requests
import os
from PIL import Image
from io import BytesIO
import pandas as pd

def testText(nearText,client):
    # I am fetching top "response_limit" results for the user
    # You can also analyse the result in a better way by taking a look at res.
    # Try printing res in the terminal and see what all contents it has.
    # used this for hybrid search params https://weaviate.io/developers/weaviate/search/hybrid

    #get collection
    collection = client.collections.get("HybridSearchExample")

    # Perform the hybrid search
    res = collection.query.hybrid(
        query=nearText,  # The model provider integration will automatically vectorize the query
        fusion_type= hp.fusion_alg,
        # max_vector_distance=hp.max_vector_distance,
        auto_limit=hp.autocut_jumps,
        limit=hp.response_limit,
        alpha=hp.query_alpha,
        return_metadata=MetadataQuery(score=True, explain_score=True),
        query_properties=["caption", "camera", "host", "job", "vsn", "plugin", "zone", "project", "address"], #Keyword search properties
        # bm25_operator=hp.keyword_search_params,
        vector=HybridVector.near_text(
            query=nearText,
            move_away=Move(force=hp.avoid_concepts_force, concepts=hp.concepts_to_avoid), #can this be used as guardrails?
            # distance=hp.max_vector_distance,
            # certainty=hp.near_text_certainty,
        ),
        rerank=Rerank(
            prop="caption", # The property to rerank on
            query=nearText  # If not provided, the original query will be used
        )
    )

    # init
    objects = []

    # Log the results
    logging.debug("============RESULTS======================")

    # Extract results from QueryReturn object type
    for obj in res.objects:
        #log results
        logging.debug("----------------%s----------------", obj.uuid)
        logging.debug(f"Properties: {obj.properties}")
        logging.debug(f"Score: {obj.metadata.score}")
        logging.debug(f"Explain Score: {obj.metadata.explain_score}")
        logging.debug(f"Rerank Score: {obj.metadata.rerank_score}")

        # Append the relevant object data into the list
        objects.append({
            "uuid": str(obj.uuid),
            "filename": obj.properties.get("filename", ""),
            "caption": obj.properties.get("caption", ""),
            "score": obj.metadata.score,
            "explainScore": obj.metadata.explain_score,
            "rerank_score": obj.metadata.rerank_score,
            "vsn": obj.properties.get("vsn", ""),
            "camera": obj.properties.get("camera", ""),
            "project": obj.properties.get("project", ""),
            "timestamp": obj.properties.get("timestamp", ""),
            "link": obj.properties.get("link", ""),
            "host": obj.properties.get("host", ""),
            "job": obj.properties.get("job", ""),
            "plugin": obj.properties.get("plugin", ""),
            "task": obj.properties.get("task", ""),
            "zone": obj.properties.get("zone", ""),
            "node": obj.properties.get("node", ""),
            "address": obj.properties.get("address", ""),
            "location_lat": get_location_coordinate(obj, "latitude"),
            "location_lon": get_location_coordinate(obj, "longitude"),
        })

    logging.debug("==============END========================")

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(objects)

    # Return the DataFrame
    return df

# TODO: how will this be implemented in a hybrid search approach? create a caption for the image entered by the user?
# def testImage(nearImage,client):
#     # I am fetching top 3 results for the user, we can change this by making small 
#     # altercations in this function and in upload.html file
#     # # adding a threshold: https://weaviate.io/developers/weaviate/search/similarity#set-a-similarity-threshold
#     imres = client.query.get("ClipExample", ["text", "_additional {certainty} "]).with_near_image(nearImage).do()
#     return {
#         "objects": ((imres['data']['Get']['ClipExample'][0]['text']),(imres['data']['Get']['ClipExample'][1]['text']),(imres['data']['Get']['ClipExample'][2]['text'])),
#         "scores": (imres['data']['Get']['ClipExample'][0]['_additional'],imres['data']['Get']['ClipExample'][1]['_additional'],imres['data']['Get']['ClipExample'][2]['_additional'])
#     }

def get_location_coordinate(obj, coordinate_type):
    """ Helper function to safely fetch latitude or longitude from the location property. """
    location = obj.properties.get("location", "")
    if location:
        try:
            # Ensure the coordinate_type is valid and fetch the correct value
            return float(getattr(location, coordinate_type, "0.0")) if coordinate_type in ["latitude", "longitude"] else "0.0"
        except (AttributeError, ValueError):
            logging.warning(f"Invalid {coordinate_type} value found for obj {obj.uuid}")
            return "0.0"  # Default fallback for invalid location
    return "0.0"  # Default fallback if location is missing

def getImage(url):
    '''
    Retrieve the Images from Sage
    '''
    #Creds
    USER = os.environ.get("SAGE_USER")
    PASS = os.environ.get("SAGE_PASS")

    # Auth header for Sage
    auth = (USER, PASS)

    try:
        # Get the image data
        response = requests.get(url, auth=auth)
        response.raise_for_status()  # Raise error for bad responses
        image_data = response.content

        # Convert the image data to a PIL Image
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")  # Ensure it's in RGB mode if necessary

    except requests.exceptions.HTTPError as e:
        logging.debug(f"Image skipped, HTTPError for URL {url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logging.debug(f"Image skipped, request failed for URL {url}: {e}")
        return None
    except Exception as e:
        logging.debug(f"Image skipped, an error occurred for URL {url}: {e}")
        return None

    return image