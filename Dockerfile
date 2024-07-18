FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04
# FROM nvidia/cudagl:11.1.1-devel-ubuntu18.04

# Define the user
ARG USER_NAME
ARG USER_ID
ARG USER_GID

# Create the user
RUN groupadd --gid $USER_GID $USER_NAME && \
    useradd --uid $USER_ID --gid $USER_GID -m $USER_NAME

# Because NVIDIA rotated public keys 
# https://forums.developer.nvidia.com/t/invalid-public-key-for-cuda-apt-repository/212901/10
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
# Therfore adding:
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install system dependencies
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y htop \
                                               curl \
                                               apt-utils \
                                               python3-dev \
                                               python3-venv \
                                               python3-pip \
                                               python3-setuptools \
                                               python3-tk \
                                               wget \
                                               mesa-utils \
                                               gcc \
                                               unzip \
                                               ffmpeg \
                                               && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install python dependencies
RUN python3 -m pip install gym==0.17.3 \
                           pybullet>=3.0.4 \
                           matplotlib>=3.1.1 \
                           opencv-python>=4.1.2.30 \
                           meshcat>=0.0.18 \
                           transforms3d==0.4.1 \
                           hydra-core==1.0.5 \
                           tdqm \
                           transformers==4.3.2 \
                           kornia==0.5.11 \
                           imageio \
                           imageio-ffmpeg

# Set pythonpath
RUN echo "export PYTHONPATH=$PYTHONPATH:/home/${USER_NAME}/docker_volume" >> /home/${USER_NAME}/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:/home/${USER_NAME}/shared_docker_volume" >> /home/${USER_NAME}/.bashrc

RUN mkdir /home/${USER_NAME}/docker_volume
RUN mkdir /home/${USER_NAME}/shared_docker_volume
WORKDIR /home/${USER_NAME}/docker_volume
USER ${USER_NAME}
CMD [ "/bin/bash" ]
