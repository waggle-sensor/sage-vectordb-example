'''Main File'''
#NOTE: This will be deployed in our cloud under k8s namespace beehive-sage
#   maybe integrated with sage-data-loader. Keep in mind, I will have to
#   somehow make the data loader not wait on creating an object in weaviate
#   because this takes longer.

import logging
import os
from client import initialize_weaviate_client
import tritonclient.grpc as TritonClient
from data import continual_load
from apscheduler.schedulers.background import BackgroundScheduler

USER = os.environ.get("SAGE_USER")
PASS = os.environ.get("SAGE_PASS")

def run_continual_load():
    '''
    Run the continual loading function in the background
    '''
    #init weaviate client
    weaviate_client = initialize_weaviate_client()

    # Initiate Triton client
    triton_client = TritonClient.InferenceServerClient(url="florence2:8001")

    # Start continual loading
    continual_load(USER, PASS, weaviate_client, triton_client)

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
    scheduler.add_job(run_continual_load)

    # Start the scheduler to run jobs in the background
    scheduler.start()