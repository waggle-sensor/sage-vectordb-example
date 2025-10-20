'''Metrics Client for sending metrics to the metrics server'''

import requests
import logging
import time
from typing import Optional

class MetricsClient:
    """Client for sending metrics to the metrics server via HTTP API"""
    
    def __init__(self, metrics_server_url='http://localhost:8080'):
        self.metrics_server_url = metrics_server_url
        self.logger = logging.getLogger(__name__)
    
    def _send_metric(self, endpoint: str, data: dict) -> bool:
        """Send metric data to the metrics server"""
        try:
            url = f"{self.metrics_server_url}/api/metrics/{endpoint}"
            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 200:
                self.logger.debug(f"[METRICS-CLIENT] Successfully sent metric to {endpoint}")
                return True
            else:
                self.logger.warning(f"[METRICS-CLIENT] Failed to send metric to {endpoint}: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"[METRICS-CLIENT] Error sending metric to {endpoint}: {e}")
            return False
    
    def record_task_processed(self, task_type: str, status: str, duration: float = 0):
        """Record a processed task"""
        data = {
            'task_type': task_type,
            'status': status,
            'duration': duration
        }
        return self._send_metric('task', data)
    
    def record_task_retry(self, task_type: str, retry_reason: str):
        """Record a task retry"""
        data = {
            'task_type': task_type,
            'retry_reason': retry_reason
        }
        return self._send_metric('task/retry', data)
    
    def record_task_duration(self, task_type: str, duration: float):
        """Record task processing duration"""
        data = {
            'task_type': task_type,
            'duration': duration
        }
        return self._send_metric('task/duration', data)
    
    def record_sage_image(self, node_id: str, camera: str):
        """Record SAGE image received"""
        data = {
            'node_id': node_id,
            'camera': camera
        }
        return self._send_metric('sage', data)
    
    def record_error(self, component: str, error_type: str):
        """Record an error"""
        data = {
            'component': component,
            'error_type': error_type
        }
        return self._send_metric('error', data)
    
    def record_model_inference(self, model_name: str, operation: str, duration: float, status: str):
        """Record model inference"""
        data = {
            'model_name': model_name,
            'operation': operation,
            'duration': duration,
            'status': status
        }
        return self._send_metric('model', data)
    
    def record_weaviate_operation(self, operation: str, status: str, duration: float):
        """Record Weaviate operation"""
        data = {
            'operation': operation,
            'status': status,
            'duration': duration
        }
        return self._send_metric('weaviate', data)
    
    def record_dlq_archive(self, error_type: str = 'general'):
        """Record DLQ archive"""
        data = {
            'error_type': error_type
        }
        return self._send_metric('dlq/archive', data)
    
    def record_dlq_reprocess(self, status: str):
        """Record DLQ reprocess"""
        data = {
            'status': status
        }
        return self._send_metric('dlq/reprocess', data)
    
    def update_queue_size(self, queue_name: str, size: int):
        """Update queue size"""
        data = {
            'queue_name': queue_name,
            'size': size
        }
        return self._send_metric('queue', data)
    
    def update_dlq_size(self, size: int):
        """Update DLQ size"""
        data = {
            'size': size
        }
        return self._send_metric('dlq/size', data)
    
    def update_active_workers(self, count: int):
        """Update active worker count"""
        data = {
            'count': count
        }
        return self._send_metric('workers', data)
    
    def record_worker_start(self, worker_id: str):
        """Record worker start time"""
        data = {
            'worker_id': worker_id
        }
        return self._send_metric('worker/start', data)
    
    def update_worker_uptime(self, worker_id: str):
        """Update worker uptime"""
        data = {
            'worker_id': worker_id
        }
        return self._send_metric('worker/uptime', data)
    
    def update_sage_stream_health(self, healthy: bool):
        """Update SAGE stream health"""
        data = {
            'healthy': healthy
        }
        return self._send_metric('sage/health', data)
    
    def update_error_rate(self, component: str, rate: float):
        """Update error rate"""
        data = {
            'component': component,
            'rate': rate
        }
        return self._send_metric('error/rate', data)
    
    def update_system_health(self, healthy: bool):
        """Update system health"""
        data = {
            'healthy': healthy
        }
        return self._send_metric('system/health', data)
    
    def update_component_health(self, component: str, healthy: bool):
        """Update component health"""
        data = {
            'component': component,
            'healthy': healthy
        }
        return self._send_metric('component/health', data)
    
    def update_memory_usage(self, component: str, bytes_used: int):
        """Update memory usage"""
        data = {
            'component': component,
            'bytes_used': bytes_used
        }
        return self._send_metric('memory', data)

# Global metrics client instance
metrics_client = MetricsClient()
