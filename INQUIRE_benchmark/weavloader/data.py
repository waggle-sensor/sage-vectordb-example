'''This file contains code that adds data to weaviate using HuggingFace.
These images will be the ones with which the hybrid search will compare
the text query given by the user.'''
import weaviate
import os
import logging
import time
from datasets import load_dataset
from io import BytesIO, BufferedReader
from PIL import Image
from model import triton_gen_caption
from weaviate.classes.data import GeoCoordinate

# Load INQUIRE benchmark dataset from Hugging Face
INQUIRE_DATASET = os.environ.get("INQUIRE_DATASET",  "sagecontinuum/INQUIRE-Benchmark-small")

def load_inquire_data(weaviate_client, triton_client):
    """
    Load images from HuggingFace INQUIRE dataset into Weaviate.
    """

    # Load dataset
    dataset = load_dataset(INQUIRE_DATASET, split="test")

    # Get Weaviate collection
    collection = weaviate_client.collections.get("INQUIRE")

    for i, item in enumerate(dataset):
        try:
            # Extract metadata
            image = item["image"]  # This is the actual image (PIL.Image)
            query = item["query"]  # Text query used for ranking
            query_id = item["query_id"]  # Unique ID for the query
            relevant = item["relevant"]  # Relevance score (0 or 1)
            clip_score = item["clip_score"]  # CLIP score for ranking
            inat_id = item["inat24_image_id"]  # Image ID from iNat24
            filename = item["inat24_file_name"]  # Original filename
            supercategory = item["supercategory"]  # High-level category
            category = item["category"]  # More specific category
            iconic_group = item["iconic_group"]  # Group type (e.g., mammal)
            species_id = item["inat24_species_id"]  # Species ID
            species_name = item["inat24_species_name"]  # Species name
            location_uncertainty = item["location_uncertainty"]  # Location uncertainty
            lat, lon = item.get("latitude", None), item.get("longitude", None)  # Location
            date = item["date"]  # Date image was taken

            # Convert image to BytesIO for encoding
            image_stream = BytesIO()
            image.save(image_stream, format="JPEG")
            image_stream.seek(0)

            # Encode image for Weaviate
            buffered_stream = BufferedReader(image_stream)
            encoded_image = weaviate.util.image_encoder_b64(buffered_stream)

            # Generate caption using Florence-2
            florence_caption = triton_gen_caption(triton_client, image)

            # Prepare data for insertion into Weaviate
            data_properties = {
                "inat24_file_name": filename,
                "image": encoded_image,
                "query": query, 
                "query_id": query_id,
                "caption": florence_caption,  # Generated caption
                "relevant": relevant,
                "clip_score": clip_score,
                "inat24_image_id": inat_id,
                "supercategory": supercategory,
                "category": category,
                "iconic_group": iconic_group,
                "inat24_species_id": species_id,
                "inat24_species_name": species_name,
                "location_uncertainty": location_uncertainty,
                "date": date,
                "location": GeoCoordinate(latitude=float(lat), longitude=float(lon)) if lat and lon else None,
            }

            # Insert into Weaviate
            collection.data.insert(properties=data_properties)
            logging.debug(f'Image {filename} added to Weaviate')

        except Exception as e:
            logging.error(f"Error processing image {filename}: {e}")

    logging.debug(f"{INQUIRE_DATASET} dataset successfully loaded into Weaviate")