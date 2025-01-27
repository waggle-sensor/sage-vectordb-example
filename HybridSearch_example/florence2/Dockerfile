# Use Triton's base image with Python 3.11
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver

# built for NVIDIA Driver Release 510 or later
FROM nvcr.io/nvidia/tritonserver:22.04-py3 

# Fix missing GPG key error
RUN apt-key del 7fa2af80 \
    && curl -L -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 605C66F00D6C9793 \
    0E98404D386FA1D9 648ACFD622F3D138

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

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV MODEL_PATH=/app/Florence-2-base
ENV MODEL_VERSION=ee1f1f163f352801f3b7af6b2b96e4baaa6ff2ff

# Download Florence 2 model from Hugging Face
RUN huggingface-cli download \
  --local-dir $MODEL_PATH \
  --revision $MODEL_VERSION \
  microsoft/Florence-2-base

# Copy the application code into the container
COPY . .

# Expose Triton server ports
EXPOSE 8000 8001 8002

# Start the Triton Inference Server with the Python Backend
CMD ["tritonserver", "--model-repository=/app/models"]
