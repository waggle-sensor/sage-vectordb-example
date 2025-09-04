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
from data import continual_load
from apscheduler.schedulers.background import BackgroundScheduler
import traceback

USER = os.environ.get("SAGE_USER")
PASS = os.environ.get("SAGE_PASS")

TRITON_HOST = os.environ.get("TRITON_HOST","triton")
TRITON_PORT = os.environ.get("TRITON_PORT","8001")

def run_continual_load():
    '''
    Run the continual loading function in the background
    '''
    #init weaviate client
    weaviate_client = initialize_weaviate_client()

    # Initiate Triton client
    channel_args = [
        ("grpc.max_metadata_size", 32 * 1024),
        ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
    ]
    triton_client = TritonClient.InferenceServerClient(url=f"{TRITON_HOST}:{TRITON_PORT}",
                                                       channel_args=channel_args,
                                                       )

    # Start continual loading
    try:
        continual_load(USER, PASS, weaviate_client, triton_client)
    except Exception as e:
        logging.error(
            f"Error in continual load [{type(e).__name__}]: {e}\n{traceback.format_exc()}"
        )
        weaviate_client.close()

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