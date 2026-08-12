FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -L \
    "https://media.githubusercontent.com/media/JOne325/fast_vit/main/vit_base_patch16_224_imagenet21k_fp16.tflite" \
    -o vit_base_patch16_224_imagenet21k_fp16.tflite

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]