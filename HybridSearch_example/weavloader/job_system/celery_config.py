'''Celery Configuration for Weavloader'''

import os

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

# Task routing
task_routes = {
    'weavloader.tasks.process_image': {'queue': 'image_processing'},
    'weavloader.tasks.monitor_data_stream': {'queue': 'data_monitoring'},
}

# Retry configuration
task_default_retry_delay = 60  # 1 minute
task_max_retries = 3
task_retry_jitter = True
task_retry_backoff = True
task_retry_backoff_max = 600  # 10 minutes max delay

# Monitoring
worker_send_task_events = True
task_send_sent_event = True

# Periodic tasks (Celery Beat)
beat_schedule = {
    'cleanup-failed-tasks': {
        'task': 'weavloader.tasks.cleanup_failed_tasks',
        'schedule': 3600.0,  # Run every hour
    },
    'reprocess-dlq-tasks': {
        'task': 'weavloader.tasks.reprocess_dlq_tasks',
        'schedule': 86400.0,  # Run daily at midnight
    },
    'dlq-health-check': {
        'task': 'weavloader.tasks.dlq_health_check',
        'schedule': 1800.0,  # Run every 30 minutes
    },
}
