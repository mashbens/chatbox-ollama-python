FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Salin semua file ke image
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install flask langchain langchain-community langchain-huggingface chromadb

# Jalankan aplikasi
CMD ["python", "app.py"]