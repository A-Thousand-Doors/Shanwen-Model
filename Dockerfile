# Start from the official NVIDIA CUDA image: https://hub.docker.com/r/nvidia/cuda/tags
# Use the CUDA 12.4.0 base image with Ubuntu 22.04
# This image includes the CUDA toolkit and development libraries
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential curl git git-lfs vim \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install packaging && \
    pip install modelscope[framework] && \
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install flash-attn --no-build-isolation

# Install verl
WORKDIR /workspace
RUN git clone https://github.com/volcengine/verl.git
WORKDIR /workspace/verl
RUN pip install -e .[vllm]
WORKDIR /workspace

CMD ["bash"]