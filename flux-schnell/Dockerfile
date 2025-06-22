FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  OUTPUT_DIR=/app/outputs \
  LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 python3-pip git wget && \
  rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir \
  torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
  pip install --no-cache-dir \
  diffusers transformers accelerate huggingface_hub \
  sentencepiece protobuf pillow fastapi "uvicorn[standard]" && \
  rm -rf /root/.cache/pip

WORKDIR /app
COPY app.py .
COPY templates ./templates
COPY static ./static

RUN mkdir -p $OUTPUT_DIR
EXPOSE 7860
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

