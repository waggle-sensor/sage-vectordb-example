'''This file contains code that adds data to weaviate using HuggingFace.
These images will be the ones with which the hybrid search will compare
the text query given by the user.'''
import weaviate
import os
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from io import BytesIO, BufferedReader
from PIL import Image
from model import triton_gen_caption
from weaviate.classes.data import GeoCoordinate
from itertools import islice

# Load INQUIRE benchmark dataset from Hugging Face
INQUIRE_DATASET = os.environ.get("INQUIRE_DATASET", "sagecontinuum/INQUIRE-Benchmark-small")

# Batch size for parallel processing
IMAGE_BATCH_SIZE = int(os.environ.get("IMAGE_BATCH_SIZE", 100))

def process_batch(batch, triton_client):
    """
    Process a batch of images and return formatted data for Weaviate.
    """
    formatted_data = []
    
    for item in batch:
        try:

            if not isinstance(item, dict):
                raise TypeError(f"Expected dict, got {type(item)} - {item}")

            logging.debug(f"Processing item: {item}")  # Debugging output

            if not isinstance(item["image"], Image.Image):
                raise TypeError(f"Expected PIL.Image, got {type(item['image'])}")

            # Extract metadata
            image = item["image"]  # PIL.Image object
            query = item["query"]
            query_id = item["query_id"]
            relevant = item["relevant"]
            clip_score = item["clip_score"]
            inat_id = item["inat24_image_id"]
            filename = item["inat24_file_name"]
            supercategory = item["supercategory"]
            category = item["category"]
            iconic_group = item["iconic_group"]
            species_id = item["inat24_species_id"]
            species_name = item["inat24_species_name"]
            location_uncertainty = item["location_uncertainty"]
            lat, lon = item.get("latitude", None), item.get("longitude", None)
            date = item["date"]

            # Convert image to BytesIO for encoding
            image_stream = BytesIO()
            image.save(image_stream, format="JPEG")
            image_stream.seek(0)

            # Encode image for Weaviate
            buffered_stream = BufferedReader(image_stream)
            encoded_image = weaviate.util.image_encoder_b64(buffered_stream)

            # Generate caption using Florence-2
            florence_caption = triton_gen_caption(triton_client, image)

            # Construct data for Weaviate
            data_properties = {
                "inat24_file_name": filename,
                "image": encoded_image,
                "query": query, 
                "query_id": query_id,
                "caption": florence_caption,
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

            formatted_data.append(data_properties)

        except Exception as e:
            logging.error(f"Error processing image {filename}: {e}")

    return formatted_data

def batched(iterable, batch_size):
    """
    Yield successive batch_size chunks from iterable.
    Args:
        iterable: An iterable (e.g., list, DataFrame rows)
        batch_size: Size of each batch
    Yields:
        list: A batch of items from the iterable
    """
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch

def load_inquire_data(weaviate_client, triton_client, sample_size=0, workers=0):
    """
    Load images from HuggingFace INQUIRE dataset into Weaviate using batch import.
    Uses parallel processing to maximize CPU usage.
    Args:
        weaviate_client: Weaviate client instance.
        triton_client: Triton client instance for image captioning.
        sample_size: Number of samples to load from the dataset (0 for all).
        workers: Number of parallel workers (0 for all available CPU cores).
    Returns:
        None
    """

    # Load dataset
    dataset = load_dataset(INQUIRE_DATASET, split="test")

    # sample the dataset if sample_size is provided
    if sample_size > 0:
        sampled_indices = random.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(sampled_indices)
    
    # if workers is not provided, use all available CPU cores
    if workers == 0:
        workers = os.cpu_count()

    # Get Weaviate collection
    collection = weaviate_client.collections.get("INQUIRE")

    # Parallel processing setup
    with ThreadPoolExecutor(max_workers=workers) as executor:

        # Process the dataset in batches
        futures = []
        for batch in batched(dataset.iterrows(), IMAGE_BATCH_SIZE):

            # Convert the batch into a list of row-wise dictionaries
            # batch_dicts = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

            # Submit the batch for processing
            futures.append(executor.submit(process_batch, batch, triton_client))

        # Batch insert into Weaviate
        # Weaviate will configure its own batch size here
        with collection.batch.dynamic() as batch:
            for future in as_completed(futures):
                formatted_data = future.result()
                if formatted_data:
                    for data_row in formatted_data:
                        batch.add_object(properties=data_row)

                    # Stop batch import if too many errors occur
                    if batch.number_errors > 5:
                        logging.error("Batch import stopped due to excessive errors.")
                        break

        # Log failed imports
        failed_objects = collection.batch.failed_objects
        if failed_objects:
            logging.debug(f"Number of failed imports: {len(failed_objects)}")
            logging.debug(f"First failed object: {failed_objects[0]}")

    logging.debug(f"{INQUIRE_DATASET} dataset successfully loaded into Weaviate")
