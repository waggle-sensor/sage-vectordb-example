#!/bin/bash

# Start data stream monitor for weavloader
echo "Starting data stream monitor..."

# Set environment variables if not already set
export CELERY_BROKER_URL=${CELERY_BROKER_URL:-"redis://localhost:6379/0"}
export CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND:-"redis://localhost:6379/0"}

# Start the monitor
python main.py monitor
