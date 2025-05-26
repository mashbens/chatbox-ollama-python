FROM python:3.12.3-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Salin semua file ke image
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install \
      flask \
      langchain \
      langchain-community \
      langchain-huggingface \
      langchain-ollama \
      chromadb \
      torch==2.7.0 \
      transformers==4.52.3

# Jalankan aplikasi
CMD ["python", "app.py"]