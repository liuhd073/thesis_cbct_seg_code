FROM nvidia/cuda:10.1-devel-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

# CUDA 10.2-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    "pytorch=1.6.0=py3.8_cuda10.1.243_cudnn7.6.3_0" \
    "torchvision=0.7.0=py38_cu101" \
 && conda clean -ya

# RUN sudo apt-get install libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx

RUN pip install tensorboard
RUN pip install simpleitk neptune-client pillow==6.2.1 scikit-image scipy opencv-python

# Set up Nvidia Apex
WORKDIR /home/user
# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /home/user/apex
RUN pip install -v --no-cache-dir --global-option="--cuda_ext" .
WORKDIR /app

# Set the default command to python3
CMD ["/bin/bash"]