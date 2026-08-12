import os
import time

print("=== APPLICATION STARTING ===", flush=True)
print(f"PORT={os.environ.get('PORT')}", flush=True)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

print("=== FASTAPI IMPORTED ===", flush=True)

import numpy as np
from PIL import Image

print("=== NUMPY/PIL IMPORTED ===", flush=True)

from tensorflow.lite.python.interpreter import Interpreter

print("=== TFLITE IMPORTED ===", flush=True)


MODEL_PATH = "vit_base_patch16_224_imagenet21k_fp16.tflite"

print("=== CREATING FASTAPI APP ===", flush=True)

app = FastAPI()

print("=== MODEL FILE DEBUG ===", flush=True)
print("Current directory:", os.getcwd(), flush=True)
print("Model path:", MODEL_PATH, flush=True)
print("Absolute path:", os.path.abspath(MODEL_PATH), flush=True)
print("Exists:", os.path.exists(MODEL_PATH), flush=True)

if os.path.exists(MODEL_PATH):
    print("Size:", os.path.getsize(MODEL_PATH), "bytes", flush=True)

    with open(MODEL_PATH, "rb") as f:
        data = f.read(32)

    print("First 32 bytes:", repr(data), flush=True)
    print("First 4 bytes:", repr(data[4:8]), flush=True)

print("=== END MODEL FILE DEBUG ===", flush=True)

print("=== LOADING TFLITE MODEL ===", flush=True)

interpreter = Interpreter(
    model_path=MODEL_PATH
)

print("=== LOADING TFLITE MODEL ===", flush=True)

interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=2
)

print("=== ALLOCATING TENSORS ===", flush=True)

interpreter.allocate_tensors()

print("=== TENSORS ALLOCATED ===", flush=True)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"INPUT: {input_details}", flush=True)
print(f"OUTPUT: {output_details}", flush=True)

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

class_names = [
    "Blackheads",
    "Cyst",
    "Papules",
    "Pustules",
    "Whiteheads"
]


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "ViT-B16-21K",
        "message": "FastAPI TFLite server is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    start = time.perf_counter()

    contents = await file.read()

    image = Image.open(
        __import__("io").BytesIO(contents)
    ).convert("RGB")

    image = image.resize((224, 224))

    image = np.asarray(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_index, image)

    inference_start = time.perf_counter()

    interpreter.invoke()

    inference_latency_ms = (
        time.perf_counter() - inference_start
    ) * 1000

    predictions = interpreter.get_tensor(output_index)[0]

    predicted_index = int(np.argmax(predictions))

    total_latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "class": class_names[predicted_index],
        "class_index": predicted_index,
        "confidence": float(predictions[predicted_index]),
        "inference_latency_ms": round(inference_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2)
    }