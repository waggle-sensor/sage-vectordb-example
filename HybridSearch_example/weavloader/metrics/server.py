'''Prometheus Metrics Server for Weavloader'''

from flask import Flask, Response, request, jsonify
from .metrics import get_metrics, metrics
import logging
import threading
import time
import psutil
import redis
from datetime import datetime

app = Flask(__name__)

@app.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(get_metrics(), mimetype='text/plain')

@app.route('/health')
def health_endpoint():
    """Health check endpoint"""
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}

@app.route('/api/metrics/task', methods=['POST'])
def record_task_metric():
    """Record task processing metrics"""
    try:
        data = request.get_json()
        task_type = data.get('task_type')
        status = data.get('status')
        duration = data.get('duration', 0)
        
        if task_type and status:
            metrics.record_task_processed(task_type, status)
            if duration > 0:
                metrics.record_task_duration(task_type, duration)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing task_type or status'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error recording task metric: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/task/retry', methods=['POST'])
def record_task_retry():
    """Record task retry metrics"""
    try:
        data = request.get_json()
        task_type = data.get('task_type')
        retry_reason = data.get('retry_reason')
        
        if task_type and retry_reason:
            metrics.record_task_retry(task_type, retry_reason)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing task_type or retry_reason'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error recording task retry: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/task/duration', methods=['POST'])
def record_task_duration():
    """Record task processing duration"""
    try:
        data = request.get_json()
        task_type = data.get('task_type')
        duration = data.get('duration')
        
        if task_type is not None and duration is not None:
            metrics.record_task_duration(task_type, duration)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing task_type or duration'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error recording task duration: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/sage', methods=['POST'])
def record_sage_metric():
    """Record SAGE image metrics"""
    try:
        data = request.get_json()
        node_id = data.get('node_id')
        camera = data.get('camera')
        
        if node_id and camera:
            metrics.record_sage_image(node_id, camera)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing node_id or camera'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error recording SAGE metric: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/error', methods=['POST'])
def record_error_metric():
    """Record error metrics"""
    try:
        data = request.get_json()
        component = data.get('component')
        error_type = data.get('error_type')
        
        if component and error_type:
            metrics.record_error(component, error_type)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing component or error_type'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error recording error metric: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/model', methods=['POST'])
def record_model_metric():
    """Record model inference metrics"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        operation = data.get('operation')
        duration = data.get('duration', 0)
        status = data.get('status')
        
        if model_name and operation and status:
            metrics.record_model_inference(model_name, operation, duration, status)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing required fields'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error recording model metric: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/weaviate', methods=['POST'])
def record_weaviate_metric():
    """Record Weaviate operation metrics"""
    try:
        data = request.get_json()
        operation = data.get('operation')
        status = data.get('status')
        duration = data.get('duration', 0)
        
        if operation and status:
            metrics.record_weaviate_operation(operation, status, duration)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing operation or status'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error recording Weaviate metric: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/dlq/archive', methods=['POST'])
def record_dlq_archive():
    """Record DLQ archive metrics"""
    try:
        data = request.get_json()
        error_type = data.get('error_type', 'general')
        
        metrics.record_dlq_archive(error_type)
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"[METRICS] Error recording DLQ archive: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/dlq/reprocess', methods=['POST'])
def record_dlq_reprocess():
    """Record DLQ reprocess metrics"""
    try:
        data = request.get_json()
        status = data.get('status')
        
        if status:
            metrics.record_dlq_reprocess(status)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing status'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error recording DLQ reprocess: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/queue', methods=['POST'])
def update_queue_metric():
    """Update queue size metrics"""
    try:
        data = request.get_json()
        queue_name = data.get('queue_name')
        size = data.get('size')
        
        if queue_name is not None and size is not None:
            metrics.update_queue_size(queue_name, size)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing queue_name or size'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error updating queue metric: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/dlq/size', methods=['POST'])
def update_dlq_size():
    """Update DLQ size metrics"""
    try:
        data = request.get_json()
        size = data.get('size')
        
        if size is not None:
            metrics.update_dlq_size(size)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing size'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error updating DLQ size: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/workers', methods=['POST'])
def update_workers_metric():
    """Update active workers count"""
    try:
        data = request.get_json()
        count = data.get('count')
        
        if count is not None:
            metrics.update_active_workers(count)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing count'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error updating workers metric: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/worker/start', methods=['POST'])
def record_worker_start():
    """Record worker start time"""
    try:
        data = request.get_json()
        worker_id = data.get('worker_id')
        
        if worker_id:
            metrics.record_worker_start(worker_id)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing worker_id'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error recording worker start: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/worker/uptime', methods=['POST'])
def update_worker_uptime():
    """Update worker uptime"""
    try:
        data = request.get_json()
        worker_id = data.get('worker_id')
        
        if worker_id:
            metrics.update_worker_uptime(worker_id)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing worker_id'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error updating worker uptime: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/sage/health', methods=['POST'])
def update_sage_stream_health():
    """Update SAGE stream health"""
    try:
        data = request.get_json()
        healthy = data.get('healthy')
        
        if healthy is not None:
            metrics.update_sage_stream_health(healthy)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing healthy'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error updating SAGE stream health: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/error/rate', methods=['POST'])
def update_error_rate():
    """Update error rate"""
    try:
        data = request.get_json()
        component = data.get('component')
        rate = data.get('rate')
        
        if component is not None and rate is not None:
            metrics.update_error_rate(component, rate)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing component or rate'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error updating error rate: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/system/health', methods=['POST'])
def update_system_health():
    """Update system health"""
    try:
        data = request.get_json()
        healthy = data.get('healthy')
        
        if healthy is not None:
            metrics.update_system_health(healthy)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing healthy'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error updating system health: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/component/health', methods=['POST'])
def update_component_health():
    """Update component health"""
    try:
        data = request.get_json()
        component = data.get('component')
        healthy = data.get('healthy')
        
        if component is not None and healthy is not None:
            metrics.update_component_health(component, healthy)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing component or healthy'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error updating component health: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/memory', methods=['POST'])
def update_memory_usage():
    """Update memory usage"""
    try:
        data = request.get_json()
        component = data.get('component')
        bytes_used = data.get('bytes_used')
        
        if component is not None and bytes_used is not None:
            metrics.update_memory_usage(component, bytes_used)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Missing component or bytes_used'}), 400
    except Exception as e:
        logging.error(f"[METRICS] Error updating memory usage: {e}")
        return jsonify({'error': str(e)}), 500

def collect_system_metrics():
    """Collect system metrics in background"""
    while True:
        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            metrics.update_memory_usage('total', memory_info.used)
            
            # Process memory for current process
            process = psutil.Process()
            metrics.update_memory_usage('worker', process.memory_info().rss)
            
            # Redis connection health and queue sizes
            try:
                redis_client = redis.Redis(host='localhost', port=6379, db=0)
                redis_client.ping()
                metrics.update_component_health('redis', True)
                
                # Get queue sizes
                image_queue_size = redis_client.llen('image_processing')
                monitor_queue_size = redis_client.llen('data_monitoring')
                cleanup_queue_size = redis_client.llen('cleanup')
                
                metrics.update_queue_size('image_processing', image_queue_size)
                metrics.update_queue_size('data_monitoring', monitor_queue_size)
                metrics.update_queue_size('cleanup', cleanup_queue_size)
                
                # Get DLQ size
                dlq_keys = redis_client.keys('dlq:*')
                metrics.update_dlq_size(len(dlq_keys))
                
            except Exception as e:
                metrics.update_component_health('redis', False)
                logging.warning(f"[METRICS] Redis connection failed: {e}")
            
            # System health (basic check)
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            system_healthy = cpu_percent < 90 and memory_percent < 90
            metrics.update_system_health(system_healthy)
            
            # Update active workers (approximate)
            active_workers_count = len([p for p in psutil.process_iter(['name']) if 'celery' in p.info['name'].lower()])
            metrics.update_active_workers(active_workers_count)
            
            # Component health checks
            # TODO: ping weaviate, triton and sage to check if they are healthy
            # NOTE: has to be moved to file with client access
            metrics.update_component_health('weaviate', True)  # Assume healthy if we reach here
            metrics.update_component_health('triton', True)    # Assume healthy if we reach here
            metrics.update_component_health('sage', True)      # Assume healthy if we reach here
            # # Check component health
            # try:
            #     # Test Weaviate connection
            #     weaviate_client.collections.list()
            #     metrics_client.update_component_health('weaviate', True)
            # except Exception:
            #     metrics_client.update_component_health('weaviate', False)
            
            # try:
            #     # Test Triton connection
            #     triton_client.is_server_ready()
            #     metrics_client.update_component_health('triton', True)
            # except Exception:
            #     metrics_client.update_component_health('triton', False)
            
            logging.debug(f"[METRICS] System metrics collected - CPU: {cpu_percent}%, Memory: {memory_percent}%, Workers: {active_workers_count}")
            
        except Exception as e:
            logging.error(f"[METRICS] Error collecting system metrics: {e}")
        
        time.sleep(30)  # Collect every 30 seconds

def start_metrics_server(port=8080):
    """Start the metrics server"""
    logging.info(f"[METRICS] Starting metrics server on port {port}")
    
    # Start background metrics collection
    metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
    metrics_thread.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=port, debug=False)
