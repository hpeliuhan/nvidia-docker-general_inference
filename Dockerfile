FROM nvcr.io/nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev libgl1  libsm6 libxext6  && \
    pip3 install --upgrade pip && \
    pip3 install numpy

# Copy the ResNet code to the container

# Set the working directory
WORKDIR /app
RUN pip3 install --upgrade setuptools wheel pip
RUN pip3 install opencv-python==4.7.0.68
#RUN pip3 install opencv-python
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torch-tensorrt==1.2.0 --no-index --find-links https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.2.0
RUN pip3 install nvidia-pyindex
RUN pip3 install nvidia-tensorrt


ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=asia/singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get  update &&  apt-get install -y libglib2.0-0 ffmpeg
# Start the ResNet script


RUN pip3 install matplotlib ipywidgets --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org
RUN apt-get install -y xorg
ENV DISPLAY=:0
RUN apt-get install wget
RUN wget  -O /app/imagenet_class_index.json "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
COPY main.py /app/main.py
COPY config /app/config
CMD ["python3","main.py","config"]
