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
import time

def start_monitor():
    """
    Start the data stream monitoring task
    """
    try:
        # Submit the monitoring task to run in the background
        monitor_data_stream.apply_async(queue="data_monitoring")
        logging.info("[MAIN] Data stream monitoring task submitted")
    except Exception as e:
        logging.error(f"[MAIN] Error starting monitor: {e}")

if __name__ == "__main__":
    # Configure logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Check what type of worker we should start
    if len(sys.argv) > 1 and sys.argv[1] == "processor":
        # Run as Celery processor worker
        logging.info("[MAIN] Starting Celery processor worker...")
        celery_app.worker_main([
            'worker',
            f'--loglevel={LOG_LEVEL.lower()}',
            '--queues=image_processing',
            '--concurrency=3',
            f'-n processor@%h'
        ])
    elif len(sys.argv) > 1 and sys.argv[1] == "moderator":
        # Run as Celery moderator worker
        logging.info("[MAIN] Starting Celery moderator worker...")
        celery_app.worker_main([
            'worker',
            f'--loglevel={LOG_LEVEL.lower()}',
            '--queues=data_monitoring',
            '--concurrency=1',
            f'-n moderator@%h'
        ])
    elif len(sys.argv) > 1 and sys.argv[1] == "cleaner":
        # Start the Celery cleanup worker
        logging.info("[MAIN] Starting Celery cleanup worker...")
        celery_app.worker_main([
            'worker',
            f'--loglevel={LOG_LEVEL.lower()}',
            '--queues=cleanup',
            '--concurrency=2',
            f'-n cleaner@%h'
        ])
    else: 
        # Start the data stream monitor as default worker
        start_monitor()
        
        # Keep running to submit tasks
        try:
            while True:
                time.sleep(60)  # Check every minute
        except (KeyboardInterrupt, SystemExit):
            logging.info("[MAIN] Monitor stopped")