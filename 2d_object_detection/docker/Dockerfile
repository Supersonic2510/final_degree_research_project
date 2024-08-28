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

# Set the default command to run when starting the container
CMD ["/bin/bash"]
