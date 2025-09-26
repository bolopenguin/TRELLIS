FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential htop git python3-onnx rdfind
RUN pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html
RUN pip install spconv-cu118
RUN pip install pipelime-python
RUN pip install easydict rembg onnxruntime transformers open3d plyfile xatlas pyvista pymeshfix igraph flash-attn==2.7.3 wheel
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@d790d337638ce478972da01e42d7ee255796f7b4
RUN git clone https://github.com/NVlabs/nvdiffrast; cd nvdiffrast; pip install .
WORKDIR /app
COPY . /app/
RUN pip install weights/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl
RUN pip install weights/diffoctreerast-0.0.0-cp311-cp311-linux_x86_64.whl
RUN mkdir -p /app/weights/hub
COPY ./weights/hub/ /root/.cache/torch/hub/
RUN pip install gsplat==1.5.1+pyt241cu118 -f http://wheels.eyecan.ai:8000 --trusted-host wheels.eyecan.ai
RUN pip install eyesplat -f http://wheels.eyecan.ai:8000 --trusted-host wheels.eyecan.ai
RUN pip install eyecan-calibry -f http://wheels.eyecan.ai:8000  --trusted-host wheels.eyecan.ai
RUN pip install cutie -f http://wheels.eyecan.ai:8000  --trusted-host wheels.eyecan.ai
COPY ./weights/checkpoints/ /root/.cache/torch/hub/checkpoints/
RUN pip install .

