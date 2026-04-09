FROM python:3.12-slim

WORKDIR /app

# build-essential provides g++ which chroma-hnswlib needs to compile its
# C++ HNSW index. Python 3.12 has pre-built wheels for most packages so
# this is only a fallback, but it's required for the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in a separate layer so Docker cache is reused
# on code-only changes (requirements.txt rarely changes, app/ changes often)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Persistent storage lives at /app/storage — mounted as a volume in docker-compose
# so chroma_db and document_registry.json survive container restarts.
RUN mkdir -p /app/storage

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
