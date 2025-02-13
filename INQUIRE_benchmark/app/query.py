'''This file implements functions that fetch results from weaviate for the query 
entered by user.'''

import HyperParameters as hp
from weaviate.classes.query import MetadataQuery, Move, HybridVector, Rerank
import logging
import requests
import os
from PIL import Image
from io import BytesIO
import pandas as pd

def testText(nearText,client):
    # used this for hybrid search params https://weaviate.io/developers/weaviate/search/hybrid

    #get collection
    collection = client.collections.get("INQUIRE")

    # Perform the hybrid search
    res = collection.query.hybrid(
        query=nearText,  # The model provider integration will automatically vectorize the query
        fusion_type= hp.fusion_alg,
        # max_vector_distance=hp.max_vector_distance,
        auto_limit=hp.autocut_jumps,
        limit=hp.response_limit,
        alpha=hp.query_alpha,
        return_metadata=MetadataQuery(score=True, explain_score=True),
        query_properties=["caption"], #Keyword search properties
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
            "inat24_image_id": obj.properties.get("inat24_image_id", ""),
            "inat24_file_name": obj.properties.get("inat24_file_name", ""),
            "score": obj.metadata.score,
            "explainScore": obj.metadata.explain_score,
            "rerank_score": obj.metadata.rerank_score,
            "query": obj.properties.get("query", ""),
            "query_id": obj.properties.get("query_id", ""),
            "caption": obj.properties.get("caption", ""),
            "relevant": obj.properties.get("relevant", ""),
            "clip_score": obj.properties.get("clip_score", ""),
            "supercategory": obj.properties.get("supercategory", ""),
            "category": obj.properties.get("category", ""),
            "iconic_group": obj.properties.get("iconic_group", ""),
            "inat24_species_id": obj.properties.get("inat24_species_id", ""),
            "inat24_species_name": obj.properties.get("inat24_species_name", ""),
            "location_uncertainty": obj.properties.get("location_uncertainty", ""),
            "date": obj.properties.get("date", ""),
            "location_lat": get_location_coordinate(obj, "latitude"),
            "location_lon": get_location_coordinate(obj, "longitude"),
        })

    logging.debug("==============END========================")

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(objects)

    # Return the DataFrame
    return df

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