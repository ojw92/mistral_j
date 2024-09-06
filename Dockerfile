# Use an existing NVIDIA CUDA image as the base
# 11.2.0 image was removed
# FROM nvidia/cuda:11.0.3-base-ubuntu20.04
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# bitsandbytes 0.43.3 requires CUDA 12.1
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Set the timezone (optional)
ENV TZ=America/New_York

# Install essential packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    apt-utils \
    curl \
    git \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter Notebook
RUN pip3 install jupyter notebook

# Copy the requirements.txt file to the working directory and install required Python packages
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
RUN pip3 install git+https://github.com/huggingface/transformers
RUN pip3 install git+https://github.com/run-llama/llama_index
RUN pip3 install git+https://github.com/TimDettmers/bitsandbytes.git
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install chromadb==0.53
RUN pip3 install pysqlite3-binary
RUN pip3 install llama-index-readers-web
RUN pip3 install llama-index-llms-huggingface
RUN pip3 install llama-index-embeddings-huggingface
RUN pip3 install llama-index-agent-openai



# Set the working directory
WORKDIR /app

# Expose Jupyter Notebook port
EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
