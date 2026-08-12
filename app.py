import io
import time

import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "vit_base_patch16_224_imagenet21k_fp16.tflite"

IMAGE_SIZE = (224, 224)

NUM_THREADS = 4

CLASS_NAMES = [
    "Blackheads",
    "Cyst",
    "Papules",
    "Pustules",
    "Whiteheads",
]


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="ViT-B16-21K FP16 API",
    description="Skin lesion classification using a FP16 TFLite ViT model.",
    version="1.0.0",
)


# ============================================================
# Load TFLite model
# ============================================================

print("Loading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=NUM_THREADS,
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("TFLite model loaded successfully.")

print("Input shape :", input_details[0]["shape"])
print("Input dtype :", input_details[0]["dtype"])
print("Output shape:", output_details[0]["shape"])
print("Output dtype:", output_details[0]["dtype"])


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepare an uploaded image for the TFLite model.

    IMPORTANT:
    The original Keras model contains:

        Rescaling(1./255)

    Therefore we DO NOT divide by 255 here.

    The pipeline is:

        Uploaded image
            ↓
        RGB
            ↓
        224 x 224
            ↓
        float32
            ↓
        TFLite
            ↓
        Rescaling(1./255)
            ↓
        ViT
    """

    # Ensure 3-channel RGB
    image = image.convert("RGB")

    # Resize to ViT input resolution
    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    # Convert to float32.
    #
    # Values remain in the original 0-255 range.
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    # Add batch dimension:
    #
    # (224, 224, 3)
    #       ↓
    # (1, 224, 224, 3)
    image = np.expand_dims(
        image,
        axis=0,
    )

    return image


# ============================================================
# Model inference
# ============================================================

def run_inference(image: Image.Image):
    """
    Run one image through the TFLite model.

    Returns:
        predicted class
        confidence
        probabilities
        inference latency
    """

    input_tensor = preprocess_image(image)

    # --------------------------------------------------------
    # Measure model inference only
    # --------------------------------------------------------

    start_time = time.perf_counter()

    interpreter.set_tensor(
        input_index,
        input_tensor,
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_index,
    )

    end_time = time.perf_counter()

    inference_latency_ms = (
        end_time - start_time
    ) * 1000.0

    # --------------------------------------------------------
    # Process predictions
    # --------------------------------------------------------

    # Output shape:
    #
    # (1, 5)
    #
    # Remove batch dimension:
    #
    # (5,)
    probabilities = output[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    # Map class names to probabilities
    class_probabilities = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "class_index": predicted_index,
        "confidence": confidence,
        "probabilities": class_probabilities,
        "inference_latency_ms": round(
            inference_latency_ms,
            3,
        ),
    }


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health_check():
    """
    Simple health endpoint.

    Used to verify that:
    - FastAPI is running
    - TFLite model loaded successfully
    """

    return {
        "status": "ok",
        "model": MODEL_PATH,
        "input_shape": input_details[0]["shape"].tolist(),
        "output_shape": output_details[0]["shape"].tolist(),
        "classes": CLASS_NAMES,
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):
    """
    Classify an uploaded image.
    """

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Could not determine file type.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # ----------------------------------------------------
        # Decode image
        # ----------------------------------------------------

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # ----------------------------------------------------
        # Run model
        # ----------------------------------------------------

        result = run_inference(image)

        # Add filename to response
        result["filename"] = file.filename

        return result

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"Inference error: {type(e).__name__}: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process image.",
        )