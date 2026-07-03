# Cloud reasoning HTTP server for Alibaba Simple Application Server (SAS) — or any Docker host.
# Serves diagnose_defect over HTTP (/healthz, /diagnose) via cloud/handler.py.
# The reasoning path is pure stdlib + urllib (calls Qwen-VL over HTTPS), so the image is tiny
# and needs no ONNX/cv2/torch — those are edge-only.

FROM python:3.11-slim

WORKDIR /app

# Copy only what the server needs: the cloud package + the edge package it imports
# (edge.dotenv for local .env loading). Datasets, models, tests, docs are NOT included.
COPY cloud/ ./cloud/
COPY edge/ ./edge/

# No third-party deps required for the reasoning HTTP server (verified: stdlib + urllib only).
# If you later add deps to the server path, create cloud/sas_deploy/requirements.txt and:
#   COPY cloud/sas_deploy/requirements.txt . && RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8080
EXPOSE 8080

# DASHSCOPE_API_KEY / QWEN_MODEL / QWEN_BASE_URL are injected at runtime (--env-file .env).
CMD ["python", "-m", "cloud.handler"]
