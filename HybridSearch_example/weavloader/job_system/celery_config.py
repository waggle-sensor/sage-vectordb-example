'''Celery Configuration for Weavloader'''

import os
from kombu import Queue

# Task queues
task_queues = (
    Queue('data_monitoring'),
    Queue('image_processing'),
    Queue('cleanup')
)
task_routes = {
    'job_system.tasks.monitor_data_stream':  {'queue': 'data_monitoring'},
    'job_system.tasks.process_image_task':   {'queue': 'image_processing'},
    'job_system.tasks.cleanup_failed_tasks': {'queue': 'cleanup'},
    'job_system.tasks.reprocess_dlq_tasks':  {'queue': 'cleanup'},
    'job_system.tasks.dlq_health_check':     {'queue': 'cleanup'},
}

# Periodic tasks (Celery Beat)
beat_schedule = {
    'cleanup-failed-tasks': {
        'task': 'job_system.tasks.cleanup_failed_tasks',
        'schedule': 900.0,  # Every hour (3600 seconds), debug: every 15 minutes (900 seconds)
        'options': {'queue': 'cleanup'},
    },
    'reprocess-dlq-tasks': {
        'task': 'job_system.tasks.reprocess_dlq_tasks',
        'schedule': 1800.0,  # Daily (86400 seconds = 24 hours), debug: every 30 minutes (1800 seconds)
        'options': {'queue': 'cleanup'},
    },
    'dlq-health-check': {
        'task': 'job_system.tasks.dlq_health_check',
        'schedule': 300.0,  # Every 30 minutes (1800 seconds), debug: every 5 minutes (300 seconds)
        'options': {'queue': 'cleanup'},
    },
}

# Celery configuration
broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Task settings
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True
worker_cancel_long_running_tasks_on_connection_loss = True

# Retry settings
task_acks_late = True
worker_prefetch_multiplier = 1
task_reject_on_worker_lost = True
broker_connection_retry_on_startup = True
broker_connection_retry = True

# Retry configuration
task_default_retry_delay = 60  # 1 minute
task_max_retries = 3
task_retry_jitter = True
task_retry_backoff = True
task_retry_backoff_max = 600  # 10 minutes max delay

# Monitoring
worker_send_task_events = True
task_send_sent_event = True
