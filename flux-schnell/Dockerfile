FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  MODEL_ID=black-forest-labs/FLUX.1-schnell \
  OUTPUT_DIR=/app/outputs \
  BATCH_SIZE=6 \
  BATCH_WAIT=0.05 \
  REQ_TIMEOUT=30 \
  MAX_QUEUE=256

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  git curl wget libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
  git+https://github.com/huggingface/diffusers.git \
  git+https://github.com/huggingface/transformers.git \
  accelerate

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools \
  && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p ${OUTPUT_DIR}

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
