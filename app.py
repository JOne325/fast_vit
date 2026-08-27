import os
import time
import io
import shutil
import glob

print("=== APPLICATION STARTING ===", flush=True)
print(f"PORT={os.environ.get('PORT')}", flush=True)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

print("=== FASTAPI IMPORTED ===", flush=True)

import numpy as np
from PIL import Image

print("=== NUMPY/PIL IMPORTED ===", flush=True)

from tensorflow.lite.python.interpreter import Interpreter

print("=== TFLITE IMPORTED ===", flush=True)


# ==========================================
# MODEL CONFIGURATION
# ==========================================

MODEL_PATH = "vit_base_patch16_224_imagenet21k_fp16.tflite"

class_names = [
    "Blackheads",
    "Cyst",
    "Papules",
    "Pustules",
    "Whiteheads"
]


# ==========================================
# CREATE FASTAPI APP
# ==========================================

print("=== CREATING FASTAPI APP ===", flush=True)

app = FastAPI()

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allow requests from the Flutter web admin dashboard (any origin for now).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# TFLITE MODEL LOADER & STATE
# ==========================================

interpreter = None
input_index = None
output_index = None
active_model_name = "ViT-B16-21K"

def load_tflite_model(path: str, model_name: str = "ViT-B16-21K"):
    """Loads and allocates tensors for a TFLite model file."""
    global interpreter, input_index, output_index, active_model_name

    print(f"=== LOADING TFLITE MODEL FROM {path} ===", flush=True)
    new_interpreter = Interpreter(
        model_path=path,
        num_threads=2
    )
    new_interpreter.allocate_tensors()

    new_input_details = new_interpreter.get_input_details()
    new_output_details = new_interpreter.get_output_details()

    input_index = new_input_details[0]["index"]
    output_index = new_output_details[0]["index"]
    interpreter = new_interpreter
    active_model_name = model_name

    print(f"=== TENSORS ALLOCATED FOR {model_name} (input={input_index}, output={output_index}) ===", flush=True)


def _run_prediction(interp, in_idx, out_idx, image_bytes: bytes):
    """
    Runs a single prediction using the given interpreter + indices.
    Returns a dict with class, confidence, and latency metrics.
    """
    start = time.perf_counter()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    interp.set_tensor(in_idx, image_array)

    inference_start = time.perf_counter()
    interp.invoke()
    inference_latency_ms = (time.perf_counter() - inference_start) * 1000

    predictions = interp.get_tensor(out_idx)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    total_latency_ms = (time.perf_counter() - start) * 1000

    # Build full probability distribution
    all_predictions = {
        class_names[i]: round(float(predictions[i]), 4)
        for i in range(min(len(predictions), len(class_names)))
    }

    return {
        "class": class_names[predicted_index] if predicted_index < len(class_names) else f"class_{predicted_index}",
        "class_index": predicted_index,
        "confidence": round(confidence, 4),
        "all_predictions": all_predictions,
        "inference_latency_ms": round(inference_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
    }


# Load default model at startup if present
if os.path.exists(MODEL_PATH):
    try:
        load_tflite_model(MODEL_PATH, "ViT-B16-21K (Default)")
    except Exception as e:
        print(f"Error loading initial model: {e}", flush=True)


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "active_model": active_model_name,
        "message": "FastAPI TFLite server is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": interpreter is not None
    }


# ── Model Info ────────────────────────────────────────────────────────────────
@app.get("/model-info")
def model_info():
    """Returns details about the currently active model and server status."""
    input_shape = None
    output_shape = None
    if interpreter is not None:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]["shape"].tolist()
        output_shape = output_details[0]["shape"].tolist()

    return {
        "status": "ok",
        "model_loaded": interpreter is not None,
        "active_model": active_model_name,
        "class_names": class_names,
        "num_classes": len(class_names),
        "input_shape": input_shape,
        "output_shape": output_shape,
    }


# ── List Uploaded Models ──────────────────────────────────────────────────────
@app.get("/models")
def list_models():
    """Lists all model files available in the models/ directory."""
    models_dir = "models"
    if not os.path.isdir(models_dir):
        return {"models": [], "count": 0}

    files = []
    for f in sorted(os.listdir(models_dir)):
        path = os.path.join(models_dir, f)
        if os.path.isfile(path):
            files.append({
                "file_name": f,
                "file_size": os.path.getsize(path),
                "modified_at": os.path.getmtime(path),
            })

    return {"models": files, "count": len(files)}


# ── Dynamic Model Upload Endpoint from Admin Dashboard ────────────────────────
@app.post("/upload-model")
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(None),
    version: str = Form(None)
):
    """
    Receives a new model file (.tflite) uploaded from the Admin Dashboard,
    saves it to disk, reloads the active TFLite interpreter, and runs a
    quick self-assessment to validate the model loads correctly.
    """
    try:
        os.makedirs("models", exist_ok=True)
        save_filename = file.filename or "uploaded_model.tflite"
        save_path = os.path.join("models", save_filename)

        # Write uploaded file to disk
        contents = await file.read()
        with open(save_path, "wb") as buffer:
            buffer.write(contents)

        display_name = f"{name or save_filename} (v{version or '1.0'})"

        # Dynamically reload interpreter
        load_tflite_model(save_path, display_name)

        # Quick self-assessment: verify the model can run inference
        # by creating a synthetic grey test image
        assessment = None
        try:
            test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            buf = io.BytesIO()
            test_image.save(buf, format="JPEG")
            assessment = _run_prediction(
                interpreter, input_index, output_index, buf.getvalue()
            )
            assessment["assessment_type"] = "self_test"
            assessment["note"] = (
                "Auto-assessment with a synthetic grey image. "
                "Use /assess-model with a real test image for meaningful results."
            )
        except Exception as assess_err:
            print(f"Self-assessment warning (non-fatal): {assess_err}", flush=True)
            assessment = {
                "assessment_type": "self_test",
                "status": "warning",
                "message": f"Model loaded but self-test failed: {str(assess_err)}",
            }

        return {
            "status": "success",
            "message": f"Model '{display_name}' uploaded and loaded successfully.",
            "file_name": save_filename,
            "active_model": active_model_name,
            "assessment": assessment,
        }
    except Exception as e:
        print(f"Failed to upload and load model: {e}", flush=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to load uploaded model: {str(e)}"
            }
        )


# ── Assess Model ──────────────────────────────────────────────────────────────
@app.post("/assess-model")
async def assess_model(
    file: UploadFile = File(None),
    test_image: UploadFile = File(...),
    name: str = Form(None),
    version: str = Form(None),
    activate: bool = Form(False),
):
    """
    Assesses an AI model by running a test prediction.

    - If `file` is provided, loads that model temporarily (or permanently
      if `activate=True`) and runs the test image through it.
    - If `file` is omitted, uses the currently active model.

    Returns the full prediction results including class probabilities,
    confidence, and latency metrics.
    """
    try:
        test_image_bytes = await test_image.read()

        if file is not None:
            # Save the model temporarily
            os.makedirs("models", exist_ok=True)
            save_filename = file.filename or "assess_model.tflite"
            save_path = os.path.join("models", save_filename)

            model_contents = await file.read()
            with open(save_path, "wb") as buffer:
                buffer.write(model_contents)

            display_name = f"{name or save_filename} (v{version or '1.0'})"

            # Load into a temporary interpreter for assessment
            temp_interpreter = Interpreter(model_path=save_path, num_threads=2)
            temp_interpreter.allocate_tensors()
            temp_input = temp_interpreter.get_input_details()[0]["index"]
            temp_output = temp_interpreter.get_output_details()[0]["index"]

            input_shape = temp_interpreter.get_input_details()[0]["shape"].tolist()
            output_shape = temp_interpreter.get_output_details()[0]["shape"].tolist()

            result = _run_prediction(
                temp_interpreter, temp_input, temp_output, test_image_bytes
            )

            # Activate the model if requested
            if activate:
                load_tflite_model(save_path, display_name)
                result["activated"] = True
                result["message"] = (
                    f"Model '{display_name}' assessed and activated as the active model."
                )
            else:
                result["activated"] = False
                result["message"] = (
                    f"Model '{display_name}' assessed successfully. "
                    "Not activated — send activate=true to make it active."
                )

            result["model_name"] = display_name
            result["file_name"] = save_filename
            result["input_shape"] = input_shape
            result["output_shape"] = output_shape

        else:
            # Assess the currently active model
            if interpreter is None:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "error",
                        "message": "No model is currently loaded on the server."
                    }
                )

            input_shape = interpreter.get_input_details()[0]["shape"].tolist()
            output_shape = interpreter.get_output_details()[0]["shape"].tolist()

            result = _run_prediction(
                interpreter, input_index, output_index, test_image_bytes
            )
            result["model_name"] = active_model_name
            result["activated"] = True
            result["message"] = (
                f"Assessment completed on active model '{active_model_name}'."
            )
            result["input_shape"] = input_shape
            result["output_shape"] = output_shape

        result["status"] = "success"
        result["assessment_type"] = "test_image"
        return result

    except Exception as e:
        print(f"Assessment failed: {e}", flush=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Assessment failed: {str(e)}"
            }
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No model loaded on server."}
        )

    contents = await file.read()
    result = _run_prediction(interpreter, input_index, output_index, contents)
    result["active_model"] = active_model_name

    return result
