'''Prometheus Metrics for Weavloader'''

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import time
import logging
from prometheus_client.multiprocess import MultiProcessCollector

# === TASK METRICS ===
# Counters for task processing
tasks_processed_total = Counter(
    'weavloader_tasks_processed_total',
    'Total number of tasks processed',
    ['task_type', 'status'],  # status: success, failure, retry
)

tasks_retried_total = Counter(
    'weavloader_tasks_retried_total',
    'Total number of task retries',
    ['task_type', 'retry_reason'],
)

# Task processing duration
task_processing_duration = Histogram(
    'weavloader_task_processing_duration_seconds',
    'Time spent processing tasks',
    ['task_type'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# === QUEUE METRICS ===
# Queue sizes
queue_size = Gauge(
    'weavloader_queue_size',
    'Current queue size',
    ['queue_name'],
    multiprocess_mode='mostrecent'
)

# DLQ metrics
dlq_size = Gauge(
    'weavloader_dlq_size',
    'Dead letter queue size',
    multiprocess_mode='mostrecent'
)

dlq_tasks_archived_total = Counter(
    'weavloader_dlq_tasks_archived_total',
    'Total tasks archived to DLQ',
    ['error_type'],
)

dlq_tasks_reprocessed_total = Counter(
    'weavloader_dlq_tasks_reprocessed_total',
    'Total tasks reprocessed from DLQ',
    ['status'],  # status: success, failure
)

# === SYSTEM METRICS ===
# Worker metrics
active_workers = Gauge(
    'weavloader_active_workers',
    'Number of active workers',
    multiprocess_mode='livemostrecent'
)

worker_uptime = Gauge(
    'weavloader_worker_uptime_seconds',
    'Worker uptime in seconds',
    ['worker_id'],
    multiprocess_mode='all'
)

# Memory usage
memory_usage_bytes = Gauge(
    'weavloader_memory_usage_bytes',
    'Memory usage in bytes',
    ['component'],  # component: worker, redis, total
    multiprocess_mode='all'
)

# === DATA STREAM METRICS ===
# SAGE data stream metrics
sage_images_received_total = Counter(
    'weavloader_sage_images_received_total',
    'Total images received from SAGE',
    ['node_id', 'camera'],
)

sage_stream_health = Gauge(
    'weavloader_sage_stream_health',
    'SAGE stream health (1=healthy, 0=unhealthy)',
    multiprocess_mode='mostrecent'
)

# === AI MODEL METRICS ===
# Model inference metrics
model_inference_duration = Histogram(
    'weavloader_model_inference_duration_seconds',
    'Time spent on model inference',
    ['model_name', 'operation'],  # operation: caption, embedding
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

model_inference_total = Counter(
    'weavloader_model_inference_total',
    'Total model inference calls',
    ['model_name', 'operation', 'status'],
)

# === DATABASE METRICS ===
# Weaviate operations
weaviate_operations_total = Counter(
    'weavloader_weaviate_operations_total',
    'Total Weaviate operations',
    ['operation', 'status'],  # operation: insert, query, delete
)

weaviate_operation_duration = Histogram(
    'weavloader_weaviate_operation_duration_seconds',
    'Time spent on Weaviate operations',
    ['operation'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# === ERROR METRICS ===
# Error rates
error_rate = Gauge(
    'weavloader_error_rate',
    'Current error rate (0-1)',
    ['component'],
    multiprocess_mode='all'
)

errors_total = Counter(
    'weavloader_errors_total',
    'Total number of errors',
    ['component', 'error_type'],
)

# === HEALTH METRICS ===
# System health
system_health = Gauge(
    'weavloader_system_health',
    'Overall system health (1=healthy, 0=unhealthy)',
    multiprocess_mode='mostrecent'
)

component_health = Gauge(
    'weavloader_component_health',
    'Component health status',
    ['component'],  # component: redis, weaviate, triton, sage
    multiprocess_mode='mostrecent'
)

# === APPLICATION INFO ===
#NOTE: Info doesn't work with multiprocess mode
# app_info = Info(
#     'weavloader_app_info',
#     'Application information',
# )

# # Set application info
# app_info.info({
#     'version': '1.0.0',
#     'description': 'Weavloader - Distributed Image Processing Service',
#     'python_version': '3.11'
# })

class MetricsCollector:
    """Metrics collection and management for Weavloader"""
    
    def __init__(self):
        self.start_time = time.time()
        self.worker_start_times = {}
        
    def record_task_processed(self, task_type: str, status: str):
        """Record a processed task"""
        tasks_processed_total.labels(task_type=task_type, status=status).inc()
        logging.debug(f"[METRICS] Task processed: {task_type} - {status}")
    
    def record_task_retry(self, task_type: str, retry_reason: str):
        """Record a task retry"""
        tasks_retried_total.labels(task_type=task_type, retry_reason=retry_reason).inc()
        logging.debug(f"[METRICS] Task retried: {task_type} - {retry_reason}")
    
    def record_task_duration(self, task_type: str, duration: float):
        """Record task processing duration"""
        task_processing_duration.labels(task_type=task_type).observe(duration)
        logging.debug(f"[METRICS] Task duration: {task_type} - {duration:.2f}s")
    
    def update_queue_size(self, queue_name: str, size: int):
        """Update queue size"""
        queue_size.labels(queue_name=queue_name).set(size)
        logging.debug(f"[METRICS] Queue size: {queue_name} - {size}")
    
    def update_dlq_size(self, size: int):
        """Update DLQ size"""
        dlq_size.set(size)
        logging.debug(f"[METRICS] DLQ size: {size}")
    
    def record_dlq_archive(self, error_type: str):
        """Record DLQ archive"""
        dlq_tasks_archived_total.labels(error_type=error_type).inc()
        logging.debug(f"[METRICS] DLQ archive: {error_type}")
    
    def record_dlq_reprocess(self, status: str):
        """Record DLQ reprocess"""
        dlq_tasks_reprocessed_total.labels(status=status).inc()
        logging.debug(f"[METRICS] DLQ reprocess: {status}")
    
    def update_active_workers(self, count: int):
        """Update active worker count"""
        active_workers.set(count)
        logging.debug(f"[METRICS] Active workers: {count}")
    
    def record_worker_start(self, worker_id: str):
        """Record worker start time"""
        self.worker_start_times[worker_id] = time.time()
        logging.debug(f"[METRICS] Worker started: {worker_id}")
    
    def update_worker_uptime(self, worker_id: str):
        """Update worker uptime"""
        if worker_id in self.worker_start_times:
            uptime = time.time() - self.worker_start_times[worker_id]
            worker_uptime.labels(worker_id=worker_id).set(uptime)
            logging.debug(f"[METRICS] Worker uptime: {worker_id} - {uptime:.2f}s")
    
    def record_sage_image(self, node_id: str, camera: str):
        """Record SAGE image received"""
        sage_images_received_total.labels(node_id=node_id, camera=camera).inc()
        logging.debug(f"[METRICS] SAGE image: {node_id} - {camera}")
    
    def update_sage_stream_health(self, healthy: bool):
        """Update SAGE stream health"""
        sage_stream_health.set(1 if healthy else 0)
        logging.debug(f"[METRICS] SAGE stream health: {healthy}")
    
    def record_model_inference(self, model_name: str, operation: str, duration: float, status: str):
        """Record model inference"""
        model_inference_duration.labels(model_name=model_name, operation=operation).observe(duration)
        model_inference_total.labels(model_name=model_name, operation=operation, status=status).inc()
        logging.debug(f"[METRICS] Model inference: {model_name} - {operation} - {duration:.2f}s - {status}")
    
    def record_weaviate_operation(self, operation: str, status: str, duration: float):
        """Record Weaviate operation"""
        weaviate_operations_total.labels(operation=operation, status=status).inc()
        weaviate_operation_duration.labels(operation=operation).observe(duration)
        logging.debug(f"[METRICS] Weaviate operation: {operation} - {status} - {duration:.2f}s")
    
    def record_error(self, component: str, error_type: str):
        """Record an error"""
        errors_total.labels(component=component, error_type=error_type).inc()
        logging.debug(f"[METRICS] Error: {component} - {error_type}")
    
    def update_error_rate(self, component: str, rate: float):
        """Update error rate"""
        error_rate.labels(component=component).set(rate)
        logging.debug(f"[METRICS] Error rate: {component} - {rate:.3f}")
    
    def update_system_health(self, healthy: bool):
        """Update overall system health"""
        system_health.set(1 if healthy else 0)
        logging.debug(f"[METRICS] System health: {healthy}")
    
    def update_component_health(self, component: str, healthy: bool):
        """Update component health"""
        component_health.labels(component=component).set(1 if healthy else 0)
        logging.debug(f"[METRICS] Component health: {component} - {healthy}")
    
    def update_memory_usage(self, component: str, bytes_used: int):
        """Update memory usage"""
        memory_usage_bytes.labels(component=component).set(bytes_used)
        logging.debug(f"[METRICS] Memory usage: {component} - {bytes_used} bytes")

# Global metrics collector instance
metrics = MetricsCollector()

def get_metrics():
    """Get metrics in Prometheus format with multiprocess support"""
    registry = CollectorRegistry()
    MultiProcessCollector(registry)
    return generate_latest(registry)