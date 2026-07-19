# Cloud reasoning HTTP server for Alibaba Simple Application Server (SAS), or any Docker host.
# Serves diagnose_defect over HTTP (/healthz, /diagnose) via cloud/handler.py.
# The reasoning path is stdlib + urllib + tenacity (backoff), so the image stays tiny and
# needs no ONNX/cv2/torch, which are edge-only.

FROM python:3.11-slim

WORKDIR /app

# Install the server's one third-party dep (tenacity) before copying code, so the layer
# caches independently of source edits.
COPY cloud/sas_deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the server needs: the cloud package + the edge package it imports
# (edge.dotenv for local .env loading). Datasets, models, tests, docs are NOT included.
COPY cloud/ ./cloud/
COPY edge/ ./edge/

ENV PORT=8080
EXPOSE 8080

# DASHSCOPE_API_KEY / QWEN_MODEL / QWEN_BASE_URL are injected at runtime (--env-file .env).
CMD ["python", "-m", "cloud.handler"]
