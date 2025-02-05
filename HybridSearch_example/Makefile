.PHONY: up down build

# Docker Network Name
NETWORK_NAME=weaviate_network
florence_image=florence2
gradio_image=gradio-ui
ollama_image=ollama
reranker_image=reranker-transformers
imagebind_image=multi2vec-bind
weaviate_image=weaviate
weavmanage_image=weavmanage
weavloader_image=weavloader
weavmanage_vol=weavmanage_data

# Up Command
up:
	# Create Docker network
	@docker network ls | grep -q $(NETWORK_NAME) || docker network create $(NETWORK_NAME)

	# Run florence2 with GPU support and custom configuration
	docker run --gpus all --name $(florence_image) --network $(NETWORK_NAME) -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=500MB --restart=on-failure -d $(florence_image)

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

	# Run Ollama
	docker run --network $(NETWORK_NAME) -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

	# Run gradio-ui container with the network configuration
	docker run --name $(gradio_image) --network $(NETWORK_NAME) -p 7860:7860 --restart on-failure \
		-e PYTHONUNBUFFERED=1 \
		-e WEAVIATE_HOST='weaviate' \
		-e WEAVIATE_PORT='8080' \
		-e WEAVIATE_GRPC_PORT='50051' \
		-e OLLAMA_HOST='ollama' \
		-e OLLAMA_PORT='11434' \
		-e CLUSTER_FLAG='True' \
		-e SAGE_USER='$(SAGE_USER)' \
		-e SAGE_PASS='$(SAGE_TOKEN)' \
		-d $(gradio_image)

	# Pull ollama model
	docker exec -it ollama ollama pull llama3.2:latest

down:
	# Stop and remove containers
	docker stop $(florence_image) $(gradio_image) $(weavmanage_image) $(weavloader_image) $(ollama_image)
	docker rm $(florence_image) $(gradio_image) $(weavmanage_image) $(weavloader_image) $(ollama_image)

# Build custom services
build:
	# Build the Florence 2 image
	docker build -t $(florence_image) ./florence2

	# Build Gradio App image
	docker build -t $(gradio_image) ./app

	# Build Weaviate Manager
	docker build -t $(weavmanage_image) ./weavmanage

	# Build Weaviate Loader
	docker build -t $(weavloader_image) ./weavloader

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
	# Create Docker network
	@docker network ls | grep -q $(NETWORK_NAME) || docker network create $(NETWORK_NAME)

	# Run multi2vec-bind with GPU support and CUDA enabled
	docker run --gpus all -e ENABLE_CUDA=1 --name $(imagebind_image) --network $(NETWORK_NAME) -d semitechnologies/multi2vec-bind:imagebind

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

#shutdown weaviate
db_down:

	# Stop and remove containers
	docker stop $(weaviate_image) $(imagebind_image) $(reranker_image)
	docker rm $(weaviate_image) $(imagebind_image) $(reranker_image)

#USE WITH CAUTION, this completely resets the whole system losing all data
reset:

	# Stop and remove all components
	docker stop $(florence_image) $(gradio_image) $(weavmanage_image) $(weavloader_image) $(weaviate_image) $(imagebind_image) $(reranker_image)
	docker rm $(florence_image) $(gradio_image) $(weavmanage_image) $(weavloader_image) $(weaviate_image) $(imagebind_image) $(reranker_image)
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