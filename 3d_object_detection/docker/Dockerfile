FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

# Set environment variables for pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid
ENV DISPLAY=$DISPLAY
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install dependencies
RUN apt-get update && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev python3-openssl git

# Install pyenv
RUN curl https://pyenv.run | bash

RUN echo 'export PYENV_ROOT="/root/.pyenv"' >> ~/.bashrc
RUN echo 'export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

# Install Python 3.11.9 and set it as global version
RUN bash -c 'source ~/.bashrc && pyenv install 3.11.9 && pyenv global 3.11.9'

WORKDIR /workspace

CMD ["bash"]
