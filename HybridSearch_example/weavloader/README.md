# Weavloader

**Weavloader** is a distributed image processing service that continuously monitors SAGE data streams, processes images with AI models, and stores them in Weaviate for hybrid search capabilities.

## **What is Weavloader?**

Weavloader is a job processing system that:

- **Monitors SAGE data streams** for new images from environmental sensors
- **Processes images** using AI models (Gemma3, CLIP, Florence2) for captioning and embeddings
- **Stores processed data** in Weaviate vector database for hybrid search
- **Handles failures gracefully** with automatic retries and dead letter queues
- **Scales horizontally** to process thousands of images efficiently

## **Architecture**

```
SAGE Data Stream → Weavloader → AI Processing → Weaviate Database
                      ↓
                 Redis Queue (Celery)
                      ↓
              Dead Letter Queue (DLQ)
```

### **Project Structure:**

```
weavloader/
├── job_system/           # Celery job processing system
│   ├── tasks.py         # Celery tasks (image processing, monitoring, DLQ)
│   └── celery_config.py # Celery configuration
├── inference/           # AI model inference
│   ├── model.py         # Triton client functions (Gemma3, CLIP, Florence2)
│   └── HyperParameters.py # Model hyperparameters
├── metrics/             # Prometheus metrics collection
│   ├── metrics.py       # Metrics definitions and collection
│   └── server.py         # HTTP metrics server
├── data.py              # SAGE data stream processing
├── client.py            # Weaviate client initialization
├── main.py              # Application entry point
└── supervisord.conf     # Process management
```

### **Components:**

- **Job System**: Celery-based task processing with retries and DLQ
- **Inference Engine**: AI model inference (Gemma3, CLIP, Florence2, ColBERT)
- **Metrics System**: Prometheus metrics collection and monitoring
- **Data Monitor**: Watches SAGE streams for new images
- **Redis**: Message broker and task queue
- **Weaviate**: Vector database for hybrid search

## **Key Features**

### **Job Processing:**
- **Automatic Retries**: Failed jobs retry with exponential backoff (60s → 120s → 240s)
- **Persistent Queues**: Jobs survive system restarts using Redis
   - If deployed using docker/k8s, redis requires volumes to persist data between restarts.
- **Horizontal Scaling**: Easy to add more workers
- **Dead Letter Queue**: Failed jobs archived for manual inspection

### **AI Inference:**
- **Multi-Model Support**: Gemma3, CLIP, Florence2, ColBERT, ALIGN
- **Triton Integration**: High-performance model serving
- **Configurable Hyperparameters**: Easy model tuning and optimization

### **Monitoring & Observability:**
- **Production Monitoring**: Built-in health checks and metrics
- **Tagged Logging**: Comprehensive logging with component tags
- **Prometheus Integration**: Detailed metrics collection
- **Grafana Dashboards**: Real-time visualization

### **Modular Architecture:**
- **Separation of Concerns**: Clean separation between job processing, inference, and metrics
- **Easy Maintenance**: Each component can be updated independently
- **Scalable Design**: Components can be scaled separately based on demand

## **Prerequisites**

### **Required Services:**
- **Redis Server** (message broker)
- **Weaviate Database** (vector storage)
- **Triton Inference Server** (ML model serving)
- **SAGE Data Access** (environmental sensor data)

### **Environment Variables:**
```bash
# SAGE Data Access
export SAGE_USER="your_username"
export SAGE_PASS="your_password"

# ML Inference Server
export TRITON_HOST="triton"
export TRITON_PORT="8001"

# Database
export WEAVIATE_HOST="weaviate"
export WEAVIATE_PORT="8080"

# Celery Configuration
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"

# Node Filtering (Optional)
export UNALLOWED_NODES="node1,node2,node3"  # Comma-separated list of nodes to exclude
```

### **Node Filtering:**
The `UNALLOWED_NODES` environment variable allows you to exclude images from specific SAGE nodes from being processed:

- **Format**: Comma-separated list of node IDs (e.g., `"W001,W002,W003"`)
- **Case Insensitive**: Node IDs are converted to lowercase for comparison
- **Whitespace Tolerant**: Spaces around node IDs are automatically trimmed
- **Default**: Empty string (no nodes excluded)

**Example:**
```bash
# Exclude specific nodes from processing
export UNALLOWED_NODES="W001,W002,test-node"
```

## **Docker Deployment**

### **Quick Start:**
```bash
# 1. Set environment variables
export SAGE_USER="your_username"
export SAGE_PASS="your_password"

# 2. Start the entire stack
docker-compose up -d

# 3. Check logs
docker-compose logs weavloader
```

### **Container Architecture:**
- **Redis Server**: Message broker (port 6379)
- **Celery Worker**: Processes image jobs
- **Celery Monitor**: Submits jobs from SAGE streams
- **Celery Beat**: Schedules cleanup and health checks
- **Supervisor**: Manages all processes

## **Configuration**

### **Retry Settings** (`celery_config.py`):
- **Max Retries**: 3 attempts per task
- **Initial Delay**: 60 seconds
- **Backoff**: Exponential (60s → 120s → 240s)
- **Max Delay**: 10 minutes

### **Queue Configuration**:
- **`image_processing`**: Individual image processing tasks
- **`data_monitoring`**: SAGE data stream monitoring
- **`dlq`**: Dead letter queue for failed tasks

### **Scheduled Tasks**:
- **Every 30 minutes**: Health check and monitoring
- **Every hour**: Archive failed tasks to DLQ
- **Daily**: Reprocess archived tasks

## **Manual Deployment**

### **Option 1: Using Scripts**
```bash
# Start Redis
redis-server

# Start Worker (processes jobs)
./start_worker.sh

# Start Monitor (submits jobs)
./start_monitor.sh
```

### **Option 2: Direct Commands**
```bash
# Start Celery Worker
python main.py

# Start Data Monitor (separate terminal)
python main.py monitor
```

### **Option 3: Celery Commands**
```bash
# Start Worker
celery -A tasks worker --loglevel=info --queues=image_processing,data_monitoring --concurrency=4

# Start Monitor
python -c "from tasks import monitor_data_stream; monitor_data_stream.delay()"
```

## **Monitoring & Observability**

### **Log Tags:**
- **`[WORKER]`**: Image processing tasks
- **`[MONITOR]`**: Data stream monitoring
- **`[CLEANUP]`**: Dead letter queue maintenance
- **`[REPROCESS]`**: DLQ task reprocessing
- **`[HEALTH]`**: System health checks
- **`[DATA]`**: Data processing operations
- **`[MODEL]`**: ML model inference
- **`[MAIN]`**: Application lifecycle

### **Monitoring Commands:**
```bash
# View active tasks
celery -A tasks inspect active

# Check queue status
redis-cli
> LLEN image_processing
> LLEN data_monitoring
> KEYS dlq:*

# View health metrics
docker-compose logs weavloader | grep "\[HEALTH\]"
```

### **Celery Flower (Optional):**
```bash
pip install flower
celery -A tasks flower
# Access at http://localhost:5555
```

## **Dead Letter Queue (DLQ)**

### **How it Works:**
1. **Failed Tasks**: After 3 retries, tasks move to DLQ
2. **Archiving**: Tasks stored with full error context for 30 days
3. **Reprocessing**: Daily automatic retry of archived tasks
4. **Cleanup**: Very old tasks (>7 days) are removed

### **DLQ Management:**
```bash
# Check DLQ size
redis-cli
> KEYS dlq:*
> LLEN dlq:*

# View archived task details
> GET dlq:task-id-here

# Manual reprocessing
python -c "from tasks import reprocess_dlq_tasks; reprocess_dlq_tasks.delay()"
```

## **Scaling**

### **Add More Workers:**
```bash
# Start additional workers
celery -A tasks worker --loglevel=info --concurrency=2 --hostname=worker2@%h
celery -A tasks worker --loglevel=info --concurrency=2 --hostname=worker3@%h
```

### **Horizontal Scaling:**
- **Workers**: Scale based on image processing load
- **Redis**: Use Redis Cluster for high availability
- **Weaviate**: Scale Weaviate cluster for storage

## **Troubleshooting**

### **Common Issues:**

1. **Redis Connection Error**:
   ```bash
   redis-cli ping  # Should return PONG
   ```

2. **Tasks Not Processing**:
   ```bash
   # Check worker status
   celery -A tasks inspect active
   
   # Check queue lengths
   redis-cli
   > LLEN image_processing
   ```

3. **High Memory Usage**:
   ```bash
   # Reduce concurrency
   celery -A tasks worker --concurrency=2
   
   # Monitor Redis memory
   redis-cli info memory
   ```

4. **DLQ Growing Large**:
   ```bash
   # Check DLQ size
   redis-cli
   > KEYS dlq:* | wc -l
   
   # Manual cleanup
   python -c "from tasks import cleanup_failed_tasks; cleanup_failed_tasks.delay()"
   ```

### **Debug Commands:**
```bash
# Check Celery status
celery -A tasks inspect active
celery -A tasks inspect registered

# Purge queues (use with caution)
celery -A tasks purge

# Check Redis status
redis-cli info
redis-cli monitor
```

## **Advanced Configuration**

### **Custom Retry Logic:**
Edit `celery_config.py` to modify:
- Retry delays
- Max retries
- Queue routing
- Task timeouts

### **Environment-Specific Settings:**
```bash
# Development
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"

# Production
export CELERY_BROKER_URL="redis://redis-cluster:6379/0"
export CELERY_RESULT_BACKEND="redis://redis-cluster:6379/0"
```

## **Metrics & Monitoring**

### **Prometheus Metrics:**
Weavloader exposes comprehensive metrics on port 8080:

- **Task Metrics**: Processing rates, success rates, duration
- **Queue Metrics**: Queue sizes, DLQ status
- **System Metrics**: Memory usage, worker count, health status
- **SAGE Metrics**: Stream health, image reception rates
- **Model Metrics**: Inference duration, error rates
- **Error Metrics**: Error rates by component

### **Accessing Metrics:**
```bash
# View metrics endpoint
curl http://localhost:8080/metrics

# Health check
curl http://localhost:8080/health
```

### **Grafana Dashboard:**
Import the provided `grafana-dashboard.json` to visualize:
- Task processing rates and success rates
- Queue sizes and DLQ status
- System health and component status
- Memory usage and performance metrics
- SAGE data stream monitoring

### **Prometheus Configuration:**
Use the provided `prometheus.yml` to scrape metrics from:
- Weavloader (port 8080)
- Redis (port 6379)
- Weaviate (port 8080)

## **Configuration Files:**

### **Core Application:**
- **`main.py`**: Application entry point
- **`data.py`**: SAGE data stream processing
- **`client.py`**: Weaviate client initialization
- **`supervisord.conf`**: Process management
- **`Dockerfile`**: Container configuration
- **`requirements.txt`**: Python dependencies

### **Job System (`job_system/`):**
- **`tasks.py`**: Celery tasks (image processing, monitoring, DLQ management)
- **`celery_config.py`**: Celery configuration and routing

### **Inference Engine (`inference/`):**
- **`model.py`**: Triton client functions (Gemma3, CLIP, Florence2, ColBERT)
- **`HyperParameters.py`**: Model hyperparameters and prompts

### **Metrics System (`metrics/`):**
- **`metrics.py`**: Prometheus metrics definitions and collection
- **`server.py`**: HTTP metrics server

### **Monitoring & Deployment:**
- **`grafana-dashboard.json`**: Grafana dashboard configuration
- **`prometheus.yml`**: Prometheus scraping configuration
- **`kubernetes-complete.yaml`**: Kubernetes deployment manifests

## **Development Workflow**

### **Adding New Models:**
1. Add model functions to `inference/model.py`
2. Update hyperparameters in `inference/HyperParameters.py`
3. Export new functions in `inference/__init__.py`
4. Update imports in `data.py` if needed

### **Adding New Tasks:**
1. Add task functions to `job_system/tasks.py`
2. Update routing in `job_system/celery_config.py`
3. Export new tasks in `job_system/__init__.py`
4. Update supervisor configuration if needed

### **Adding New Metrics:**
1. Add metric definitions to `metrics/metrics.py`
2. Update collection logic in `metrics/server.py`
3. Export new metrics in `metrics/__init__.py`
4. Update Grafana dashboard configuration

### **Testing Components:**
```bash
# Test job system
python -c "from job_system import celery_app; print('Job system OK')"

# Test inference
python -c "from inference import gemma3_run_model; print('Inference OK')"

# Test metrics
python -c "from metrics import metrics; print('Metrics OK')"
```
