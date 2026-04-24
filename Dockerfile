FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV FACE_INDEX_DIR=/app/indexes
ENV PYTHONUNBUFFERED=1
ENV FACE_ENGINE_LISTEN_HOST=0.0.0.0
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
RUN mkdir -p /app/indexes

EXPOSE 8088
CMD ["sh", "-c", "uvicorn app.main:app --host ${FACE_ENGINE_LISTEN_HOST:-0.0.0.0} --port ${PORT:-8088}"]
