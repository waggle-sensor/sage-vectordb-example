'''Celery Tasks for Weavloader'''
import logging
import os
import traceback
from celery import Celery
import tritonclient.grpc as TritonClient
from client import initialize_weaviate_client
from data import process_single_image_data
from datetime import datetime, timedelta
from metrics import metrics
import time

# Initialize Celery app
app = Celery('weavloader')
app.config_from_object('job_system.celery_config')

# Get environment variables
USER = os.environ.get("SAGE_USER")
PASS = os.environ.get("SAGE_PASS")
TRITON_HOST = os.environ.get("TRITON_HOST", "triton")
TRITON_PORT = os.environ.get("TRITON_PORT", "8001")

@app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def process_image_task(self, image_data):
    """
    Process a single image with retry logic.
    
    Args:
        image_data (dict): Dictionary containing image metadata and URL
        
    Returns:
        dict: Processing result
    """
    start_time = time.time()
    task_type = "process_image"
    
    try:
        logging.info(f"[WORKER] Processing image: {image_data.get('url', 'unknown')}")
        
        # Initialize clients (these will be reused across tasks)
        weaviate_client = initialize_weaviate_client()
        
        # Initialize Triton client
        channel_args = [
            ("grpc.max_metadata_size", 32 * 1024),
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ]
        triton_client = TritonClient.InferenceServerClient(
            url=f"{TRITON_HOST}:{TRITON_PORT}",
            channel_args=channel_args,
        )
        
        # Process the image
        result = process_single_image_data(
            image_data, 
            USER, 
            PASS, 
            weaviate_client, 
            triton_client
        )
        
        # Close clients
        weaviate_client.close()
        
        # Record successful task
        duration = time.time() - start_time
        metrics.record_task_processed(task_type, "success")
        metrics.record_task_duration(task_type, duration)
        
        logging.info(f"[WORKER] Successfully processed image: {image_data.get('url', 'unknown')}")
        return result
        
    except Exception as exc:
        # Record failed task
        duration = time.time() - start_time
        metrics.record_task_processed(task_type, "failure")
        metrics.record_task_duration(task_type, duration)
        metrics.record_error("worker", type(exc).__name__)
        
        logging.error(f"[WORKER] Error processing image {image_data.get('url', 'unknown')}: {str(exc)}")
        logging.error(f"[WORKER] Traceback: {traceback.format_exc()}")
        
        # Retry with exponential backoff
        retry_delay = 60 * (2 ** self.request.retries)  # 60s, 120s, 240s
        metrics.record_task_retry(task_type, type(exc).__name__)
        logging.warning(f"[WORKER] Retrying in {retry_delay} seconds (attempt {self.request.retries + 1}/3)")
        
        raise self.retry(countdown=retry_delay, exc=exc)

@app.task
def monitor_data_stream():
    """
    Monitor the SAGE data stream and submit image processing tasks.
    This runs continuously and submits individual images as tasks.
    """
    from data import watch
    
    logging.info("[MONITOR] Starting data stream monitoring")
    
    # Setup filter to query specific data
    filter_config = {
        "plugin": "registry.sagecontinuum.org/yonghokim/imagesampler.*",
        # "task": "imagesampler-.*"
    }
    
    try:
        # Update SAGE stream health
        metrics.update_sage_stream_health(True)
        
        # Watch for data in real-time
        for df in watch(start=None, filter=filter_config):
            vsns = df['meta.vsn'].unique()
            end_time = df.timestamp.max()
            start_time = df.timestamp.min()
            
            logging.info(f'[MONITOR] Processing images for nodes: {vsns}')
            logging.info(f'[MONITOR] Time range: {start_time} to {end_time}')
            
            # Submit each image as a separate task
            for i in df.index:
                image_data = {
                    'url': df.value[i],
                    'timestamp': df.timestamp[i].isoformat(),
                    'vsn': df["meta.vsn"][i],
                    'filename': df["meta.filename"][i],
                    'camera': df["meta.camera"][i],
                    'host': df["meta.host"][i],
                    'job': df["meta.job"][i],
                    'node': df["meta.node"][i],
                    'plugin': df["meta.plugin"][i],
                    'task': df["meta.task"][i],
                    'zone': df["meta.zone"][i],
                }
                
                # Record SAGE image received
                metrics.record_sage_image(df["meta.vsn"][i], df["meta.camera"][i])
                
                # Submit task to Celery queue
                process_image_task.delay(image_data)
                logging.debug(f"[MONITOR] Submitted image task: {image_data['url']}")
                
    except Exception as e:
        # Update SAGE stream health
        metrics.update_sage_stream_health(False)
        metrics.record_error("monitor", type(e).__name__)
        
        logging.error(f"[MONITOR] Error in data stream monitoring: {str(e)}")
        logging.error(f"[MONITOR] Traceback: {traceback.format_exc()}")
        raise

@app.task
def cleanup_failed_tasks():
    """
    cleanup task for failed jobs - moves permanently failed tasks 
    to dead letter queue for later reprocessing.
    """
    import redis
    import json
    from celery.result import AsyncResult
    from celery import current_app
    
    # Connect to Redis with proper configuration
    redis_client = redis.Redis(
        host='localhost', 
        port=6379, 
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True
    )
    
    logging.info("[CLEANUP] Running cleanup for failed tasks")
    
    try:
        # Get tasks that have been failing for more than 1 hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        # Get all task results from Redis
        task_keys = redis_client.keys("celery-task-meta-*")
        
        archived_count = 0
        processed_count = 0
        
        for key in task_keys:
            try:
                processed_count += 1
                task_data = redis_client.get(key)
                
                if not task_data:
                    continue
                    
                task_info = json.loads(task_data)
                task_id = task_info.get('task_id')
                
                if not task_id:
                    continue
                
                # Check if task failed and is old enough
                if (task_info.get('status') == 'FAILURE' and 
                    task_info.get('date_done')):
                    
                    # Parse date properly
                    try:
                        task_date = datetime.fromisoformat(
                            task_info['date_done'].replace('Z', '+00:00')
                        )
                        
                        if task_date < cutoff_time:
                            # Get full task result for better error details
                            try:
                                result = AsyncResult(task_id, app=current_app)
                                error_details = {
                                    'exc_type': getattr(result.result, '__class__', {}).get('__name__', 'UnknownError'),
                                    'exc_message': str(result.result) if result.result else 'Unknown error',
                                    'traceback': getattr(result.result, '__traceback__', None)
                                }
                            except Exception:
                                error_details = {
                                    'exc_type': 'UnknownError',
                                    'exc_message': task_info.get('result', {}).get('exc_message', 'Unknown error'),
                                    'traceback': None
                                }
                            
                            # Archive to dead letter queue with comprehensive data
                            dlq_key = f"dlq:{task_id}"
                            archive_data = {
                                'task_id': task_id,
                                'original_task': task_info,
                                'archived_at': datetime.now().isoformat(),
                                'retry_count': task_info.get('retries', 0),
                                'error_details': error_details,
                                'task_name': task_info.get('task', 'unknown'),
                                'args': task_info.get('args', []),
                                'kwargs': task_info.get('kwargs', {}),
                                'eta': task_info.get('eta'),
                                'expires': task_info.get('expires'),
                                'priority': task_info.get('priority'),
                                'routing_key': task_info.get('routing_key')
                            }
                            
                            # Store in dead letter queue with TTL of 30 days
                            redis_client.setex(
                                dlq_key, 
                                int(timedelta(days=30).total_seconds()),
                                json.dumps(archive_data, default=str)
                            )
                            
                            # Remove from original location
                            redis_client.delete(key)
                            
                            archived_count += 1
                            logging.info(f"[CLEANUP] Archived failed task {task_id} to DLQ")
                            
                    except (ValueError, TypeError) as e:
                        logging.warning(f"[CLEANUP] Error parsing date for task {task_id}: {e}")
                        continue
                        
            except Exception as e:
                logging.warning(f"[CLEANUP] Error processing task key {key}: {e}")
                continue
        
        # Update DLQ metrics
        metrics.update_dlq_size(archived_count)
        if archived_count > 0:
            metrics.record_dlq_archive("general")
        
        # Production logging with metrics
        logging.info(f"[CLEANUP] Cleanup completed: {archived_count}/{processed_count} tasks archived to dead letter queue")
        
        # alerting for high failure rates
        if archived_count > 0:
            failure_rate = (archived_count / processed_count) * 100 if processed_count > 0 else 0
            metrics.update_error_rate("cleanup", failure_rate / 100)
            logging.warning(f"[CLEANUP] {archived_count} tasks moved to dead letter queue ({failure_rate:.1f}% failure rate)")
            
            # Alert if failure rate is too high
            if failure_rate > 10:  # More than 10% failure rate
                logging.error(f"[CLEANUP] HIGH FAILURE RATE: {failure_rate:.1f}% of tasks are failing!")
                
        # Clean up old DLQ entries (older than 30 days)
        cleanup_old_dlq_entries(redis_client)
            
    except Exception as e:
        logging.error(f"[CLEANUP] Cleanup task failed but continuing: {e}")

def cleanup_old_dlq_entries(redis_client):
    """Clean up old dead letter queue entries"""
    try:
        dlq_keys = redis_client.keys("dlq:*")
        cleaned_count = 0
        
        for dlq_key in dlq_keys:
            try:
                # Check TTL
                ttl = redis_client.ttl(dlq_key)
                if ttl == -1:  # No expiration set
                    # Set expiration for old entries without TTL
                    redis_client.expire(dlq_key, int(timedelta(days=30).total_seconds()))
                elif ttl == -2:  # Key doesn't exist
                    cleaned_count += 1
            except Exception as e:
                logging.warning(f"[CLEANUP] Error cleaning DLQ key {dlq_key}: {e}")
                
        if cleaned_count > 0:
            logging.info(f"[CLEANUP] Cleaned up {cleaned_count} expired DLQ entries")
            
    except Exception as e:
        logging.warning(f"[CLEANUP] Error in cleanup_old_dlq_entries: {e}")

@app.task
def reprocess_dlq_tasks():
    """
    Production-ready reprocessing of dead letter queue tasks.
    Retries archived failed tasks with proper error handling and rate limiting.
    """
    import redis
    import json
    
    # Connect to Redis with production settings
    redis_client = redis.Redis(
        host='localhost', 
        port=6379, 
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True
    )
    
    logging.info("[REPROCESS] Starting production reprocessing of dead letter queue tasks")
    
    try:
        # Get all DLQ tasks
        dlq_keys = redis_client.keys("dlq:*")
        
        if not dlq_keys:
            logging.info("[REPROCESS] No tasks in dead letter queue to reprocess")
            return
        
        reprocessed_count = 0
        failed_reprocess_count = 0
        skipped_count = 0
        
        # Rate limiting: process max 100 tasks per run to avoid overwhelming the system
        max_tasks_per_run = 100
        tasks_to_process = dlq_keys[:max_tasks_per_run]
        
        logging.info(f"[REPROCESS] Processing {len(tasks_to_process)} tasks from DLQ (max {max_tasks_per_run} per run)")
        
        for dlq_key in tasks_to_process:
            try:
                # Get archived task data
                archived_data = redis_client.get(dlq_key)
                if not archived_data:
                    skipped_count += 1
                    continue
                    
                task_archive = json.loads(archived_data)
                original_task = task_archive.get('original_task', {})
                
                # Check if task is too old (older than 7 days, skip to avoid infinite loops)
                archived_at = task_archive.get('archived_at')
                if archived_at:
                    try:
                        archive_date = datetime.fromisoformat(archived_at)
                        if archive_date < datetime.now() - timedelta(days=7):
                            logging.warning(f"[REPROCESS] Skipping very old DLQ task {dlq_key} (archived {archive_date})")
                            redis_client.delete(dlq_key)  # Remove very old tasks
                            skipped_count += 1
                            continue
                    except (ValueError, TypeError):
                        pass
                
                # Extract task arguments
                task_args = original_task.get('args', [])
                task_kwargs = original_task.get('kwargs', {})
                task_name = original_task.get('task', '')
                
                # Validate task data
                if not task_name or not task_args:
                    logging.warning(f"[REPROCESS] Invalid task data in DLQ {dlq_key}, removing")
                    redis_client.delete(dlq_key)
                    skipped_count += 1
                    continue
                
                # Resubmit the task based on type
                if task_name == 'weavloader.tasks.process_image_task':
                    # Resubmit image processing task with delay to avoid overwhelming
                    result = process_image_task.apply_async(
                        args=task_args, 
                        kwargs=task_kwargs,
                        countdown=30  # 30 second delay
                    )
                    logging.info(f"[REPROCESS] Resubmitted DLQ task {dlq_key} as {result.id}")
                    
                    # Remove from DLQ only after successful submission
                    redis_client.delete(dlq_key)
                    reprocessed_count += 1
                    
                else:
                    logging.warning(f"[REPROCESS] Unknown task type {task_name} in DLQ, skipping")
                    skipped_count += 1
                    
            except Exception as e:
                logging.error(f"[REPROCESS] Error reprocessing DLQ task {dlq_key}: {e}")
                failed_reprocess_count += 1
                continue
        
        # Update DLQ reprocess metrics
        if reprocessed_count > 0:
            metrics.record_dlq_reprocess("success")
        if failed_reprocess_count > 0:
            metrics.record_dlq_reprocess("failure")
        
        # Production logging with detailed metrics
        total_processed = reprocessed_count + failed_reprocess_count + skipped_count
        success_rate = (reprocessed_count / total_processed) * 100 if total_processed > 0 else 0
        
        logging.info(f"[REPROCESS] Reprocessing completed: {reprocessed_count} resubmitted, {failed_reprocess_count} failed, {skipped_count} skipped")
        logging.info(f"[REPROCESS] Success rate: {success_rate:.1f}%")
        
        # Alert if reprocessing success rate is low
        if success_rate < 50 and total_processed > 10:
            logging.error(f"[REPROCESS] LOW REPROCESSING SUCCESS RATE: {success_rate:.1f}% - investigate DLQ tasks")
        
    except Exception as e:
        logging.error(f"[REPROCESS] Error in reprocess_dlq_tasks: {e}")
        # Don't raise in production to avoid breaking the scheduler
        logging.error(f"[REPROCESS] Reprocessing task failed but continuing: {e}")

@app.task
def dlq_health_check():
    """
    Production health check for dead letter queue - monitors DLQ size and health.
    """
    import redis
    import json
    from datetime import datetime, timedelta
    
    redis_client = redis.Redis(
        host='localhost', 
        port=6379, 
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    
    try:
        # Get DLQ statistics
        dlq_keys = redis_client.keys("dlq:*")
        dlq_size = len(dlq_keys)
        
        # Analyze DLQ composition
        error_types = {}
        task_types = {}
        recent_failures = 0
        
        for dlq_key in dlq_keys:
            try:
                archived_data = redis_client.get(dlq_key)
                if archived_data:
                    task_archive = json.loads(archived_data)
                    error_details = task_archive.get('error_details', {})
                    task_name = task_archive.get('task_name', 'unknown')
                    
                    # Count error types
                    exc_type = error_details.get('exc_type', 'UnknownError')
                    error_types[exc_type] = error_types.get(exc_type, 0) + 1
                    
                    # Count task types
                    task_types[task_name] = task_types.get(task_name, 0) + 1
                    
                    # Count recent failures (last 24 hours)
                    archived_at = task_archive.get('archived_at')
                    if archived_at:
                        try:
                            archive_date = datetime.fromisoformat(archived_at)
                            if archive_date > datetime.now() - timedelta(hours=24):
                                recent_failures += 1
                        except (ValueError, TypeError):
                            pass
                            
            except Exception as e:
                logging.warning(f"Error analyzing DLQ key {dlq_key}: {e}")
                continue
        
        # Log health metrics
        logging.info(f"[HEALTH] DLQ Health Check: {dlq_size} total tasks, {recent_failures} recent failures")
        
        if error_types:
            logging.info(f"[HEALTH] Top error types: {dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        if task_types:
            logging.info(f"[HEALTH] Task types in DLQ: {task_types}")
        
        # Alert if DLQ is growing too large
        if dlq_size > 1000:
            logging.error(f"[HEALTH] DLQ SIZE WARNING: {dlq_size} tasks in dead letter queue!")
        
        # Alert if recent failure rate is high
        if recent_failures > 100:
            logging.error(f"[HEALTH] HIGH RECENT FAILURE RATE: {recent_failures} failures in last 24h")
            
    except Exception as e:
        logging.error(f"[HEALTH] Error in dlq_health_check: {e}")
