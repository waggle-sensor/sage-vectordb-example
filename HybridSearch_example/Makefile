.PHONY: up down build

# Docker Network Name
NETWORK_NAME=weaviate_network
florence_image=florence2
gradio_image=gradio-ui
reranker_image=reranker-transformers
imagehind_image=multi2vec-bind
weaviate_image=weaviate

# Up Command
up:
	# Create Docker network
	@docker network ls | grep -q $(NETWORK_NAME) || docker network create $(NETWORK_NAME)

	# Run multi2vec-bind with GPU support and CUDA enabled
	docker run --gpus all -e ENABLE_CUDA=1 --name $(imagehind_image) --network $(NETWORK_NAME) -d semitechnologies/multi2vec-bind:imagebind

	# Run florence2 with GPU support and custom configuration
	docker run --gpus all --name $(florence_image) --network $(NETWORK_NAME) -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=500MB --restart=on-failure -d $(florence_image)

	# Run reranker-transformers with GPU support and CUDA enabled
	docker run --gpus all -e ENABLE_CUDA=1 --name $(reranker_image) --network $(NETWORK_NAME) -d semitechnologies/reranker-transformers:cross-encoder-ms-marco-MiniLM-L-6-v2-1.1.1

	# Run weaviate with network configuration
	# additional env vars:
	# -e LOG_LEVEL='debug' #default is info
	# -e PROMETHEUS_MONITORING_ENABLED="true" #https://weaviate.io/developers/weaviate/configuration/monitoring 
	# -e PROMETHEUS_MONITORING_PORT=2112
	docker run --name $(weaviate_image) --network $(NETWORK_NAME) -p 8080:8080 -p 50051:50051 -d \
		-e BIND_INFERENCE_API="http://multi2vec-bind:8080" \
		-e RERANKER_INFERENCE_API="http://reranker-transformers:8080" \
		-e QUERY_DEFAULTS_LIMIT=25 \
		-e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED="true" \
		-e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
		-e DEFAULT_VECTORIZER_MODULE="multi2vec-bind" \
		-e ENABLE_MODULES="multi2vec-bind,reranker-transformers" \
		-e CLUSTER_HOSTNAME="node1" \
		-e ASYNC_INDEXING="true" \
		-e USE_BLOCKMAX_WAND="true" \
		-e USE_INVERTED_SEARCHABLE="true" \
		semitechnologies/weaviate:1.28.2

	# Run gradio-ui container with the network configuration and additional env variables
	docker run --name $(gradio_image) --network $(NETWORK_NAME) -p 7860:7860 --restart on-failure \
		-e WEAVIATE_HOST='weaviate' \
		-e WEAVIATE_PORT='8080' \
		-e WEAVIATE_GRPC_PORT='50051' \
		-e CLUSTER_FLAG='True' \
		-e CONTINUAL_LOADING=$(CONTINUAL_LOADING) \  
		-e SAGE_USER=$(SAGE_USER) \  
		-e SAGE_PASS=$(SAGE_TOKEN) \  
		-d $(gradio_image)

down:
	# Stop and remove containers
	docker stop $(imagehind_image) $(florence_image) $(reranker_image) $(weaviate_image) $(gradio_image)
	docker rm $(imagehind_image) $(florence_image) $(reranker_image) $(weaviate_image) $(gradio_image)

	# Remove the Docker network
	docker network rm $(NETWORK_NAME)

# Build custom services
build:
	# Build the Florence 2 image
	docker build -t $(florence_image) ./florence2

	# Optionally, build other images like gradio-ui if needed
	docker build -t $(gradio_image) ./app