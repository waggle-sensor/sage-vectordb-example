'''Prometheus Metrics Server for Weavloader'''

from flask import Flask, Response
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
            
            # Redis connection health
            try:
                redis_client = redis.Redis(host='localhost', port=6379, db=0)
                redis_client.ping()
                metrics.update_component_health('redis', True)
            except Exception:
                metrics.update_component_health('redis', False)
            
            # System health (basic check)
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            system_healthy = cpu_percent < 90 and memory_percent < 90
            metrics.update_system_health(system_healthy)
            
            logging.debug(f"[METRICS] System metrics collected - CPU: {cpu_percent}%, Memory: {memory_percent}%")
            
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_metrics_server()
