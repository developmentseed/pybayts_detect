FROM nvidia/opencl:runtime-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    wget \
    curl \
    libpocl-dev

RUN curl -fsSL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs

