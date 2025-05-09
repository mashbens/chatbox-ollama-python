# Gunakan Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch sesuai CPU (karena GPU perlu CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Buat working directory
WORKDIR /app

# Salin semua file ke container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan default API saat container start
CMD ["python", "app.py", "--api"]
