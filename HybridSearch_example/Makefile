.PHONY: up down build
SHELL := /bin/bash

# Docker Network Name
NETWORK_NAME=weaviate_network
triton_image=triton
gradio_image=gradio-ui
reranker_image=reranker-transformers
imagebind_image=multi2vec-bind
weaviate_image=weaviate
weavmanage_image=weavmanage
weavloader_image=weavloader
weavmanage_vol=weavmanage_data

# Up Command
up:
	# Check if env vars are set
	@if [ -z "$(SAGE_USER)" ] || [ -z "$(SAGE_TOKEN)" ] || [ -z "$(HF_TOKEN)" ] || [ -z "$(CUDA_VISIBLE_DEVICES)" ]; then \
		echo "SAGE_USER, SAGE_TOKEN, CUDA_VISIBLE_DEVICES, and HF_TOKEN must be set in the .env file."; \
		exit 1; \
	fi

	# Create Docker network
	@docker network ls | grep -q $(NETWORK_NAME) || docker network create $(NETWORK_NAME)

	# Run triton with GPU support and custom configuration
	docker run --gpus "device=$(CUDA_VISIBLE_DEVICES)" \
	-e CUDA_VISIBLE_DEVICES='$(CUDA_VISIBLE_DEVICES)' \
	--name $(triton_image) \
	--network $(NETWORK_NAME) -p 8000:8000 -p 8001:8001 -p 8002:8002 \
	--shm-size=500MB \
	--restart=on-failure \
	-d $(triton_image)

	# Create Docker volume for persistent migration data
	docker volume create $(weavmanage_vol) 

	# Run WeavManage container with the network configuration
	docker run --name $(weavmanage_image) --network $(NETWORK_NAME) --restart on-failure \
	-e WEAVIATE_HOST='weaviate' \
	-e WEAVIATE_PORT='8080' \
	-e WEAVIATE_GRPC_PORT='50051' \
	-v $(weavmanage_vol):/app/active \
	-d $(weavmanage_image)

	# Run data loader
	docker run --name $(weavloader_image) --network $(NETWORK_NAME) --restart on-failure \
		-e WEAVIATE_HOST='weaviate' \
		-e WEAVIATE_PORT='8080' \
		-e WEAVIATE_GRPC_PORT='50051' \
		-e SAGE_USER='$(SAGE_USER)' \
		-e SAGE_PASS='$(SAGE_TOKEN)' \
		-d $(weavloader_image)

	# Run gradio-ui container with the network configuration
	docker run --name $(gradio_image) --network $(NETWORK_NAME) -p 7860:7860 --restart on-failure \
		-e PYTHONUNBUFFERED=1 \
		-e WEAVIATE_HOST='weaviate' \
		-e WEAVIATE_PORT='8080' \
		-e WEAVIATE_GRPC_PORT='50051' \
		-e CLUSTER_FLAG='True' \
		-e SAGE_USER='$(SAGE_USER)' \
		-e SAGE_PASS='$(SAGE_TOKEN)' \
		-d $(gradio_image)

down:
	# Stop and remove containers
	docker stop $(triton_image) $(gradio_image) $(weavmanage_image) $(weavloader_image)
	docker rm $(triton_image) $(gradio_image) $(weavmanage_image) $(weavloader_image)

# Build custom services
build: 
	# Check if env vars are set
	@if [ -z "$(SAGE_USER)" ] || [ -z "$(SAGE_TOKEN)" ] || [ -z "$(HF_TOKEN)" ]; then \
		echo "SAGE_USER, SAGE_TOKEN, and HF_TOKEN must be set in the .env file."; \
		exit 1; \
	fi

	# Build the triton image
	docker build --build-arg HF_TOKEN='$(HF_TOKEN)' -t $(triton_image) ./triton

	# Build Gradio App image
	docker build -t $(gradio_image) ./app

	# Build Weaviate Manager
	docker build -t $(weavmanage_image) ./weavmanage

	# Build Weaviate Loader
	docker build -t $(weavloader_image) ./weavloader

	# Remove dangling images, to clean up old images
	docker image prune -f

#migrate the weaviate db
migrate:

	# Run WeavManage container with the network configuration
	docker run --name $(weavmanage_image) --network $(NETWORK_NAME) --restart on-failure \
	-e WEAVIATE_HOST='weaviate' \
	-e WEAVIATE_PORT='8080' \
	-e WEAVIATE_GRPC_PORT='50051' \
	-v $(weavmanage_vol):/app/active \
	-d $(weavmanage_image)

#deploy weaviate
db:
	# Check if env vars are set
	@if [ -z "$(CUDA_VISIBLE_DEVICES)" ] || [ -z "$(PLATFORM)" ]; then \
		echo "CUDA_VISIBLE_DEVICES and PLATFORM must be set in the .env file."; \
		exit 1; \
	fi

	# Create Docker network
	@docker network ls | grep -q $(NETWORK_NAME) || docker network create $(NETWORK_NAME)

	# Run multi2vec-bind with GPU support and CUDA enabled
	docker run --gpus "device=$(CUDA_VISIBLE_DEVICES)" \
	-e ENABLE_CUDA=1 \
	-e CUDA_CORE='cuda:$(CUDA_VISIBLE_DEVICES)' \
	--name $(imagebind_image) \
	--network $(NETWORK_NAME) \
	--platform $(PLATFORM) \
	-d semitechnologies/multi2vec-bind:imagebind-latest

	# Run reranker-transformers with GPU support and CUDA enabled
	docker run --gpus "device=$(CUDA_VISIBLE_DEVICES)" \
	-e ENABLE_CUDA=1 \
	-e CUDA_CORE='cuda:$(CUDA_VISIBLE_DEVICES)' \
	--name $(reranker_image) \
	--network $(NETWORK_NAME) \
	--platform $(PLATFORM) \
	-d semitechnologies/reranker-transformers:cross-encoder-ms-marco-MiniLM-L-6-v2-latest

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
		semitechnologies/weaviate:1.31.0

#shutdown weaviate
db_down:

	# Stop and remove containers
	docker stop $(weaviate_image) $(imagebind_image) $(reranker_image)
	docker rm $(weaviate_image) $(imagebind_image) $(reranker_image)

#USE WITH CAUTION, this completely resets the whole system losing all data
reset:

	# Stop and remove all components
	docker stop $(triton_image) $(gradio_image) $(weavmanage_image) $(weavloader_image) $(weaviate_image) $(imagebind_image) $(reranker_image)
	docker rm $(triton_image) $(gradio_image) $(weavmanage_image) $(weavloader_image) $(weaviate_image) $(imagebind_image) $(reranker_image)
	docker volume rm $(weavmanage_vol) 

	echo "The system was reset, you can now start weaviate with make db & the other components with make up"

#Regenerate the UI (mostly used to regenerate the gradio share link since its only active for 72 hours)
regen_ui:

	# Stop and remove ui container
	docker stop $(gradio_image)
	docker rm $(gradio_image)

	# Run gradio-ui container with the network configuration
	docker run --name $(gradio_image) --network $(NETWORK_NAME) -p 7860:7860 --restart on-failure \
		-e PYTHONUNBUFFERED=1 \
		-e WEAVIATE_HOST='weaviate' \
		-e WEAVIATE_PORT='8080' \
		-e WEAVIATE_GRPC_PORT='50051' \
		-e CLUSTER_FLAG='True' \
		-e SAGE_USER='$(SAGE_USER)' \
		-e SAGE_PASS='$(SAGE_TOKEN)' \
		-d $(gradio_image)