FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git wget libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv $VIRTUAL_ENV

# Upgrade pip inside venv and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
WORKDIR /app
COPY . .

ARG TARGET
RUN echo "Building target=$TARGET"

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
