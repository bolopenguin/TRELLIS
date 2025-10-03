FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

RUN apt-get update && \
    apt-get install -y ffmpeg build-essential htop git python3-onnx rdfind && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libx11-6 \
    libgl1 \
    libglu1-mesa


WORKDIR /app
COPY . /app/

RUN pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118
RUN pip install -e . -f http://wheels.eyecan.ai:8000  --trusted-host wheels.eyecan.ai
