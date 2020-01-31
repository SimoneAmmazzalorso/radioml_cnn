#Dockerfile for GPU usage
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="simone.ammazzalorso@unito.it"

#INSTALL MINICONDA
RUN apt-get update && apt-get install -y curl
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
SHELL ["/bin/bash", "-c"]

RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /Miniconda3
ENV PATH /Miniconda3/bin:$PATH

# INSTALL TENSORFLOW AND KERAS
RUN conda update -n base -c defaults conda
#RUN conda config --add channels conda-forge
RUN conda create -n tensorflow python=3.6 numpy==1.17.4 Pillow h5py

ENV PATH /Miniconda3/envs/tensorflow/bin:$PATH
RUN /bin/bash -c "source activate tensorflow" && pip uninstall tensorflow tensorflow-gpu && pip install tensorflow-gpu==1.14 keras==2.2.4 PyYAML==5.1

#
RUN mkdir /archive && mkdir /archive/home && mkdir /archive/home/sammazza && mkdir /archive/home/sammazza/radioML && mkdir /archive/home/sammazza/radioML/data && mkdir /run_CNN
WORKDIR /run_CNN
ADD CNN.py /run_CNN/CNN.py
ADD utility/image_provider.py /run_CNN/utility/image_provider.py
ADD utility/network.py /run_CNN/utility/network.py

# Overwrite the entrypoint of the base Docker image (python)
CMD ["/bin/bash","-c","/archive/home/sammazza/radioML/script.sh"]
