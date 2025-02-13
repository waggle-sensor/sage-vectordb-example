'''This file contains the code to run the Benchmark and save the results.'''

import os
from inquire_eval import evaluate_queries
from datasets import load_dataset
from client import initialize_weaviate_client
import logging
import time

# Load INQUIRE benchmark dataset from Hugging Face
INQUIRE_DATASET = os.environ.get("INQUIRE_DATASET", "sagecontinuum/INQUIRE-Benchmark-small")

def load_inquire_dataset():
    """ Load INQUIRE dataset from HuggingFace and return it as a pandas DataFrame. """
    dataset = load_dataset(INQUIRE_DATASET, split="test").to_pandas()
    return dataset

if __name__ == "__main__":
    # Connect to Weaviate
    weaviate_client = initialize_weaviate_client()

    # Load INQUIRE dataset
    inquire_dataset = load_inquire_dataset()

    # Evaluate search system
    image_results, query_evaluation = evaluate_queries(weaviate_client, inquire_dataset)

    # Save results
    image_results.to_csv("image_search_results.csv", index=False)
    query_evaluation.to_csv("query_eval_metrics.csv", index=False)
    logging.debug("Evaluation is done, INQUIRE results saved to image_search_results.csv and query_eval_metrics.csv")

    # Keep the program running when the evaluation is done
    try:
        while True:
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        exit()