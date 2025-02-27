FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    libsm6 \
    x11-utils \
    x11-apps \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements.txt to the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter Lab's default port
EXPOSE 8888

# Start Xvfb and Jupyter Lab
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --allow-root"]
