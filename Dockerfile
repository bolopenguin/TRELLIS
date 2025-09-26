FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential htop git python3-onnx rdfind && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . /app/

# Install custom wheel files (required for gaussian and octree rendering)
RUN pip install weights/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl \
    weights/diffoctreerast-0.0.0-cp311-cp311-linux_x86_64.whl

# Setup model cache directories
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    cp -r ./weights/hub/* /root/.cache/torch/hub/ && \
    cp -r ./weights/checkpoints/* /root/.cache/torch/hub/checkpoints/

# Install package with all dependencies from pyproject.toml
# Use custom index for eyecan packages
RUN pip install . -f http://wheels.eyecan.ai:8000  --trusted-host wheels.eyecan.ai

