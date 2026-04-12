FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first to avoid the large CUDA build (~2 GB)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Install the package itself
COPY . .
RUN pip install --no-cache-dir -e .

# The tutor currently runs as a one-shot CLI.
# Replace this with a uvicorn/FastAPI entrypoint when exposing as an HTTP service.
CMD ["ai-tutor"]
