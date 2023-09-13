# Use an official Ubuntu image as a parent image
FROM ubuntu:20.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Set the DEBIAN_FRONTEND environment variable to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install GPU dependencies (NVIDIA CUDA Toolkit and cuDNN)
RUN apt-get install -y --no-install-recommends \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && add-apt-repository multiverse \
    && apt-get update \
    && apt-get install -y nvidia-cuda-toolkit ffmpeg git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python3", "main.py"]




################################
# BUILD DETAILS FOR PRODUCTION #
################################
# FIRST INSTALL CUDA AND NVIDIA DRIVERS
# RUN docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
# IF IT RUNS PROPERLY AND OUTPUT IS SHOWN THEN PROCEED
# RUN docker build -t sih .