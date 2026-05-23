# Base image
FROM python:3.12-slim

# OS dependencies for MuJoCo rendering
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default execution
CMD ["python", "ppo/test.py"]