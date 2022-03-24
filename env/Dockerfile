# adapted from Paperspace/fastai-docker

FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    python-qt4 \
    libjpeg-dev \
    zip \
    unzip \
    openssh-client \
    libpng-dev &&\
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTHON_VERSION=3.9

RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build


ENV PATH=$PATH:/opt/conda/bin/
ENV USER env

COPY requirements.txt .
RUN pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt install git-lfs
RUN git lfs install

WORKDIR /notebooks
RUN chmod -R a+w /notebooks

COPY jupyter.sh /jupyter.sh
CMD ["/jupyter.sh"]













