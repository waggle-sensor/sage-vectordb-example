# I have to build my own image to change torch libs for CUDA Version: 12.4 (Sage H100s)

FROM semitechnologies/multi2vec-bind:imagebind

RUN pip install torch==2.0.1+cu124 torchvision==0.15.2+cu124 torchaudio==2.0.2 -f https://download.pytorch.org/whl/cuda/12.4/torch_stable.html