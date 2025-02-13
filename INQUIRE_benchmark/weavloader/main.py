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
from apscheduler.schedulers.background import BackgroundScheduler

def run_load():
    '''
    Run the loading function in the background
    '''
    #init weaviate client
    weaviate_client = initialize_weaviate_client()

    # Initiate Triton client
    triton_client = TritonClient.InferenceServerClient(url="florence2:8001")

    # Start loading
    load_inquire_data(weaviate_client, triton_client)

if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    # Initialize the background scheduler
    scheduler = BackgroundScheduler()

    # Schedule the continual_load function
    scheduler.add_job(run_load)

    #NOTE: I can add parallel loading of images using the scheduler, I will need to restructure the code though
    #   so that each job knows what section of images to handle
    #scheduler.add_job(run_continual_load, max_instances=2)

    # Start the scheduler to run jobs in the background
    scheduler.start()

    # Keep the program running to allow the background scheduler to continue running
    try:
        while True:
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        # Handle any exceptions to gracefully shutdown the scheduler
        scheduler.shutdown()