FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

# Set the working directory
WORKDIR /app

# Copy the entire current directory into the container
COPY . .

# Install python packages from requirements.txt
RUN pip3 install -r requirements.txt

# Set the entrypoint
# Run main.py when the container launches
CMD ["python3", "main.py"]


################################
# BUILD DETAILS FOR PRODUCTION #
################################
# FIRST INSTALL CUDA AND NVIDIA DRIVERS, NVIDIA-DOCKER-2
# RUN docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
# IF IT RUNS PROPERLY AND OUTPUT IS SHOWN THEN PROCEED
# RUN docker build -t sih .