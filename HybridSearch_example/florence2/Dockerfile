# Use Triton's base image with Python 3.11
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver

# built for NVIDIA Driver Release 510 or later (Sage Blades, V033)
FROM nvcr.io/nvidia/tritonserver:22.04-pyt-python-py3 

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

# Install dependencies using pip
RUN pip install --no-cache-dir --no-build-isolation packaging
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt
RUN pip install --no-cache-dir --force-reinstall -r torch_requirements.txt

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
