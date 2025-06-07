# Use Triton's base image with Python 3.11
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver

# built for NVIDIA Driver Release 510 or later (Sage Blades, V033)
# FROM nvcr.io/nvidia/tritonserver:22.04-py3 <-- used this when using florence 2 base model, swtiched to 23.12 for Qwen2.5-VL-72B-Instruct
FROM nvcr.io/nvidia/tritonserver:23.12-py3

# built for NVIDIA Driver Release 545 or later (Sage H100)
# FROM nvcr.io/nvidia/tritonserver:24.06-py3 
# FROM nvcr.io/nvidia/tritonserver:24.06-pyt-python-py3

# Fix missing GPG key error
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Install system dependencies
RUN apt-get update \
  && apt-get install -y \
  wget \
  curl \
  libgl1 \
  libglib2.0-0 \
  git

# Set working directory
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .
COPY torch_requirements.txt .
#COPY flash_requirements.txt . enable for flash attention, flash attention must have have CUDA 11.7 and above 

# Install dependencies using pip
RUN pip install --no-cache-dir  --no-build-isolation -r requirements.txt
RUN pip install --no-cache-dir --force-reinstall -r torch_requirements.txt
# RUN pip install --no-cache-dir --no-build-isolation -r flash_requirements.txt enable for flash attention, flash attention must have have CUDA 11.7 and above 

# Set environment variables
ENV MODEL_PATH=/app/Florence-2-base
ENV MODEL_VERSION=ee1f1f163f352801f3b7af6b2b96e4baaa6ff2ff
ENV COLBERT_MODEL_PATH=/app/colbertv2.0
ENV COLBERT_MODEL_VERSION=c1e84128e85ef755c096a95bdb06b47793b13acf
ENV ALIGN_MODEL_PATH=/app/align-base
ENV ALIGN_MODEL_VERSION=e96a37facc7b1f59090ece82293226b817afd6ba
ENV CLIP_MODEL_PATH=/app/DFN5B-CLIP-ViT-H-14-378
ENV CLIP_MODEL_VERSION=01b771ed0d1395ca5ffdd279897d665ebe00dfd2
ENV QWEN_MODEL_PATH=/app/Qwen2.5-VL-7B-Instruct-AWQ
ENV QWEN_MODEL_VERSION=536a35794df8831aa814970ee8f89eff577e7718

# Download Florence 2 model from Hugging Face
# RUN huggingface-cli download \
#   --local-dir $MODEL_PATH \
#   --revision $MODEL_VERSION \
#   microsoft/Florence-2-base

# Download ColBERT model from Hugging Face
# RUN huggingface-cli download \
#   --local-dir $COLBERT_MODEL_PATH \
#   --revision $COLBERT_MODEL_VERSION \
#   colbert-ir/colbertv2.0

# Download allign model from Hugging Face
# RUN huggingface-cli download \
#   --local-dir $ALIGN_MODEL_PATH \
#   --revision $ALIGN_MODEL_VERSION \
#   kakaobrain/align-base

# Download CLIP model from Hugging Face
RUN huggingface-cli download \
  --local-dir $CLIP_MODEL_PATH \
  --revision $CLIP_MODEL_VERSION \
  apple/DFN5B-CLIP-ViT-H-14-378

# Download Qwen model from Hugging Face
RUN huggingface-cli download \
  --local-dir $QWEN_MODEL_PATH \
  --revision $QWEN_MODEL_VERSION \
  Qwen/Qwen2.5-VL-7B-Instruct-AWQ

# Copy the application code into the container
COPY . .

# Expose Triton server ports
EXPOSE 8000 8001 8002

# Start the Triton Inference Server with the Python Backend
CMD ["tritonserver", "--model-repository=/app/models"]
