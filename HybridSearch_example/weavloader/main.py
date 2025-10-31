'''Main File'''
#NOTE: This will be deployed in our cloud under k8s namespace beehive-sage
#   maybe integrated with sage-data-loader. Keep in mind, I will have to
#   somehow make the data loader not wait on creating an object in weaviate
#   because this takes longer.

import logging
import os
import sys
from celery import Celery
from job_system import app as celery_app, monitor_data_stream
import argparse

#TODO: redo the system so that I dont need a queue just to run one continuous task
def submit_monitor_task():
    """
    Submit the data stream monitoring task
    """
    try:
        # Submit the monitoring task to run in the background
        monitor_data_stream.apply_async(queue="data_monitoring")
        logging.info("[MAIN] Data stream monitoring task submitted")
    except Exception as e:
        logging.error(f"[MAIN] Error submitting monitor task: {e}")

if __name__ == "__main__":
    # configure arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_level",
        default=os.getenv("LOG_LEVEL","INFO"),
        help="Log level.",
    )
    parser.add_argument(
        "--worker_type",
        default=None,
        help="Worker type to start (processor, moderator, cleaner).",
        choices=["processor", "moderator", "cleaner"],
    )
    args = parser.parse_args()

    # Configure logging
    LOG_LEVEL = args.log_level.upper()
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Check what type of worker we should start
    if args.worker_type == "processor":
        # Run as Celery processor worker
        logging.info("[MAIN] Starting Celery processor worker...")
        celery_app.worker_main([
            'worker',
            f'--loglevel={LOG_LEVEL.lower()}',
            '--queues=image_processing',
            '--concurrency=6',
            f'-n processor@%h'
        ])
    elif args.worker_type == "moderator":
        # Run as Celery moderator worker
        logging.info("[MAIN] Starting Celery moderator worker...")
        # submit the data stream monitor task to the data_monitoring queue
        submit_monitor_task()
        # start the moderator worker
        celery_app.worker_main([
            'worker',
            f'--loglevel={LOG_LEVEL.lower()}',
            '--queues=data_monitoring',
            '--concurrency=1',
            f'-n moderator@%h'
        ])
    elif args.worker_type == "cleaner":
        # Start the Celery cleanup worker
        logging.info("[MAIN] Starting Celery cleanup worker...")
        celery_app.worker_main([
            'worker',
            f'--loglevel={LOG_LEVEL.lower()}',
            '--queues=cleanup',
            '--concurrency=3',
            f'-n cleaner@%h'
        ])
    else: 
        logging.error("[MAIN] Invalid worker type, must be one of: processor, moderator, cleaner")
        sys.exit(1)