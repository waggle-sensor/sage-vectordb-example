# I have to build my own image to change torch libs for CUDA Version: 12.4 (Sage H100s)

FROM semitechnologies/multi2vec-bind:imagebind

# Switch to root user to avoid permission issues
USER root

RUN pip uninstall -y torch torchvision torchaudio

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu124

#switch back
USER app