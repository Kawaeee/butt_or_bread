FROM ubuntu:bionic
LABEL maintainer="kawaekc@gmail.com"

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Update apt package
RUN apt update --fix-missing

# Install required dependencies by default
RUN apt install -y wget curl git htop nano

# miniconda3 - Python 3.7
WORKDIR /opt/
ARG HOME="/opt"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b \
    && rm -r Miniconda3-py37_4.10.3-Linux-x86_64.sh
ENV PATH="/opt/miniconda3/bin:${PATH}"

# butt_or_bread repository
RUN git clone https://github.com/Kawaeee/butt_or_bread.git

# Download model
RUN cd /opt/butt_or_bread/
RUN wget https://github.com/Kawaeee/butt_or_bread/releases/download/v1.1/buttbread_resnet152_3.h5

# Install python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# House-keeping
RUN conda clean -a -y
RUN pip cache purge
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt autoclean
RUN apt autoremove

ENV HOME="/root"

ENTRYPOINT ["streamlit", "run", "/opt/butt_or_bread/streamlit_app.py"]