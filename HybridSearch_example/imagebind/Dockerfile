# I have to build my own image to change torch libs for CUDA Version: 12.4 (Sage H100s)

FROM semitechnologies/multi2vec-bind:imagebind

# Switch to root user to avoid permission issues
USER root

RUN pip uninstall -y torch

RUN pip install torch==2.4.0

#switch back
USER app