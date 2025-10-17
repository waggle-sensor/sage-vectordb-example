#!/bin/bash

# Start Celery worker for weavloader
echo "Starting Celery worker..."

# Set environment variables if not already set
export CELERY_BROKER_URL=${CELERY_BROKER_URL:-"redis://localhost:6379/0"}
export CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND:-"redis://localhost:6379/0"}

# Start the worker
python main.py
