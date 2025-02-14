'''Main File'''
#NOTE: This will be deployed in our cloud under k8s namespace beehive-sage
#   maybe integrated with sage-data-loader. Keep in mind, I will have to
#   somehow make the data loader not wait on creating an object in weaviate
#   because this takes longer.

import logging
import os
import time
from client import initialize_weaviate_client
import tritonclient.grpc as TritonClient
from data import load_inquire_data
from init import run

SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 0))
WORKERS = int(os.environ.get("WORKERS", 0))

def run_load():
    '''
    Run the loading function
    '''
    #init weaviate client
    weaviate_client = initialize_weaviate_client()

    # Initiate Triton client
    triton_client = TritonClient.InferenceServerClient(url="florence2:8001")

    # create the schema
    run(weaviate_client)

    # Start loading
    load_inquire_data(weaviate_client, triton_client, SAMPLE_SIZE, WORKERS)

    #close the client
    weaviate_client.close()

if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    # load the data into weaviate
    run_load()

    # Keep the program running when the loading is done
    try:
        while True:
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        exit()