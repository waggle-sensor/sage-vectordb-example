# FROM python:3.11-slim
FROM python:3.11.11-bullseye

RUN apt-get update \
  && apt-get install -y \
  wget \
  curl

# Set working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV MODEL_PATH=/app/Florence-2-base

# Using huggingface-cli to download the Florence2 model
# --local-dir disables the caching in huggingface-cli to place
# actual files in the --local-dir, not in $HOME/.cache/huggingface
# Also, we pin the commit that was on July 20, 2024
RUN huggingface-cli download \
  --local-dir $MODEL_PATH \
  --revision ee1f1f163f352801f3b7af6b2b96e4baaa6ff2ff \
  microsoft/Florence-2-base

# Copy the rest of the application code into the container
COPY . .

# Run upload.py
CMD ["python", "grui.py"]