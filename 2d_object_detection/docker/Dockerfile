# Use NVIDIA's CUDA base image with Python support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Use Bash as the default shell
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid
ENV DISPLAY=$DISPLAY
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Set the working directory inside the container
WORKDIR /app

# Install dependencies for pyenv and Python build
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Install Python 3.12.4 using pyenv and set it as the global version
RUN pyenv install 3.12.4 && pyenv global 3.12.4

# Install pip and other Python dependencies
RUN pip install --upgrade pip

# Copy requirements.txt to the working directory
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to run when starting the container
CMD ["/bin/bash"]
