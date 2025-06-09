FROM python:3.11-slim

WORKDIR /app

# Non-buffered output
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
