# Use Alpine Linux as the base image
FROM python:3.12.4-alpine

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install build dependencies and bash
RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev \
    build-base \
    freetype-dev \
    libpng-dev \
    openblas-dev \
    jpeg-dev \
    zlib-dev \
    bash  # Add bash to the container

# Copy requirements.txt to the working directory
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to run when starting the container
CMD ["/bin/bash"]
