FROM matthewfeickert/docker-python3-ubuntu:3.8.7
USER root
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    wget \
    curl \
    libpocl-dev \
    nodejs
