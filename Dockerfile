FROM python:3.11-slim

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY DnCNN /app/DnCNN

# Ensure upload and output directories exist
RUN mkdir -p /app/DnCNN/uploads /app/DnCNN/outputs

WORKDIR /app/DnCNN

ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --timeout 120 --workers 1"]
