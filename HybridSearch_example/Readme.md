# Hybrid Search with Florence 2 and Weaviate

This project demonstrates **Hybrid Search** where image captions are generated using **Florence 2**, and a search is conducted using both:
1. **Vector Search**: Combining the vector embeddings of both the image and its generated caption.
2. **Keyword Search**: Leveraging the captions of the images for text-based search.

The **Hybrid Search** integrates both search types into one to improve accuracy and retrieval results. After retrieving the objects, they are passed into a **reranker model** to evaluate the relevance of the results based on the context of the query, ensuring that each object is compared more effectively.

---

## Features:
- **Florence 2 for Caption Generation**: Captions are generated for images using the Florence 2 model.
- **Vector Search**: Utilizes embeddings of both the images and their captions to perform semantic search.
- **Keyword Search**: Searches are also performed using keywords extracted from image captions.
- **Hybrid Search**: A combination of vector and keyword searches to return the most relevant results.
- **Reranker**: A model that refines the order of search results, ensuring that the most relevant documents or items are ranked higher. It goes beyond the initial retrieval step, considering additional factors such as semantic similarity, context, and other relevant features.

---

## Running the Example Using Makefile
>NOTE: use this deployment if Docker Compose doesn't have access to GPU, like in node-V033

### Prerequisites
To run this example, you'll need:
- **Docker** installed on your machine with GPU access
- **Cuda** v11.6
- NVIDIA Driver Release 510 or later

### Step-by-Step Setup

1. **Spin up your Weaviate instance**:
   - Navigate to the directory containing the `Makefile` file and run:
     ```bash
     make build && make up
     ```

2. **Access Weaviate's UI**:
   - After your Weaviate instance is running, access the user interface at:
     ```
     http://localhost:7860/
     ```

3. **Image Access**:
   - Before running, make sure you have access to the image data from Sage. You will need to fetch the relevant image dataset to perform searches.

---

## Running the Example Using Docker Compose
### Prerequisites
To run this example, you'll need:
- **Docker** installed on your machine with GPU access
- **Docker Compose** v1.28.0+ with GPU access for orchestrating the multi-container application
- Basic familiarity with **docker-compose** commands
- **Cuda** v11.6
- NVIDIA Driver Release 510 or later

### Step-by-Step Setup

1. **Spin up your Weaviate instance**:
   - Navigate to the directory containing the `docker-compose.yml` file and run:
     ```bash
     docker-compose up -d
     ```
> NOTE: Docker Compose v1.28.0+ allows to define GPU reservations using the device structure defined in the Compose Specification. Lower versions will raise an error.

2. **Access Weaviate's UI**:
   - After your Weaviate instance is running, access the user interface at:
     ```
     http://localhost:7860/
     ```

3. **Image Access**:
   - Before running, make sure you have access to the image data from Sage. You will need to fetch the relevant image dataset to perform searches.

---

### Optional: **Enable Continual Loading of Images**
If you want continual loading of images, set the `CONTINUAL_LOADING` environment variable to `true` and set up your cred environment variables as well. You can either:
- Set the environment variable directly in the terminal like this:
  ```bash
  export CONTINUAL_LOADING=true
  export SAGE_USER=__INSERT_HERE__
  export SAGE_TOKEN=__INSERT_HERE__
  ```
- Or, you can create a `.env` file in the root of your project with the following content:
  ```sh
  export CONTINUAL_LOADING=true
  export SAGE_USER=__INSERT_HERE__
  export SAGE_TOKEN=__INSERT_HERE__
  ```
- Then, run the same `make` commands:
  ```bash
  source .env && make build && make up
  ```

This will start the system and continuously load images from the Sage data client. The system will check for new data and process it accordingly.

### Explanation:
When `CONTINUAL_LOADING=true`, the system will keep watching and fetching images from Sage, process them, and load them into Weaviate as they become available.

---

## Important Notes

- **Accessing the UI Remotely**:
   - If your Weaviate instance is running on a remote machine, use SSH tunneling to access the UI:
     ```bash
     ssh <client> -L 7860:<EXTERNAL-IP>:7860
     ```
   - Example:
     ```bash
     ssh node-V033 -L 7860:10.31.81.1:7860
     ```

---

## Workflow Overview

1. **Caption Generation with Florence 2**:
   - The **Florence 2** model generates captions for images, allowing for both semantic and keyword-based search.
   
2. **Vector Search**:
   - The embeddings of the images and their captions are stored in **Weaviate**. When a query is made, the relevant vectors are retrieved using similarity search (e.g., cosine similarity).

3. **Keyword Search**:
   - The captions are indexed and can be searched with keywords. This enables traditional text-based search capabilities (e.g., bm25 algorithm).

4. **Hybrid Search**:
   - A **hybrid search** combines the results from both the **vector search** and the **keyword search**. This improves result relevance by considering both semantic similarity and exact text matches.

5. **Reranking**:
   - After retrieving the results, a reranker model evaluates them against the original query. This model takes into account **context** to ensure that the most relevant and accurate results are returned.

---

## References

- **Weaviate Documentation**:  
   - [Managing Data in Weaviate](https://weaviate.io/developers/weaviate/manage-data)
   - [Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid)
   - [Hybrid Search Blog](https://weaviate.io/blog/hybrid-search-fusion-algorithms)
   - [Model Providers: ImageBind](https://weaviate.io/developers/weaviate/model-providers/imagebind/embeddings-multimodal)
   - [ImageBind Repo](https://github.com/weaviate/multi2vec-bind-inference)
   - [Reranker](https://weaviate.io/developers/weaviate/model-providers/transformers/reranker)
   - [Reranker Model Blog](https://weaviate.io/blog/ranking-models-for-better-search)
   - [Reranker Model: ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
   - [Reranker Repo v1.1.1](https://github.com/weaviate/reranker-transformers/tree/1.1.1)
   - [Vector Indexes](https://weaviate.io/developers/weaviate/config-refs/schema/vector-index)
   - [ANN Benchmarks](https://weaviate.io/developers/weaviate/benchmarks/ann)
   - [Env Variables](https://weaviate.io/developers/weaviate/config-refs/env-vars)
   

- **Triton Documentation**:
   - [Triton Server](https://github.com/triton-inference-server/server)
   - [Triton Server Tutorials](https://github.com/triton-inference-server/tutorials)
   - [Triton Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix)
   - [Triton Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
   - 
---
