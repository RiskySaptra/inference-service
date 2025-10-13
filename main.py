from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks
import uuid
from PIL import Image
import io
from ultralytics import YOLO
import torch
import shutil
from retraining_worker import run_retraining_pipeline

app = FastAPI(title="YOLOv8 Inference and Retraining API")

# --- Global State ---
# In-memory store for job statuses
retraining_jobs = {}

# Determine the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the YOLOv8 model
# Note: Replace 'your_model.pt' with the actual path to your model file.
# The model should be placed in the 'models' directory.
try:
    model = YOLO('models/v1.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- API Endpoints ---

@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"status": "API is running"}

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0),
    overlap: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Endpoint to receive an image and return YOLOv8 predictions.
    """
    if not model:
        return {"error": "Model is not loaded"}

    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Save the image
    image_path = f"inference_images/{uuid.uuid4()}.png"
    image.save(image_path)

    # Perform inference
    results = model(image, conf=confidence, iou=overlap, device=device)

    # Format the results
    predictions = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            xywh = box.xywh[0]
            
            predictions.append({
                "x": float(xywh[0]),
                "y": float(xywh[1]),
                "width": float(xywh[2]),
                "height": float(xywh[3]),
                "confidence": float(box.conf[0]),
                "class": str(int(box.cls[0])),
                "class_id": int(box.cls[0]),
                "detection_id": str(uuid.uuid4())
            })

    return {"predictions": predictions}

@app.post("/retrain/upload/")
async def upload_and_retrain(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a new dataset and start the retraining pipeline in the background.
    """
    task_id = str(uuid.uuid4())
    
    # Save the uploaded zip file
    zip_path = f"temp_{task_id}.zip"
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Start the background task
    background_tasks.add_task(run_retraining_pipeline, task_id, zip_path, retraining_jobs)

    return {"task_id": task_id, "status": "Retraining started"}

@app.get("/retrain/status/{task_id}")
def get_retraining_status(task_id: str):
    """
    Get the status of a retraining job.
    """
    status = retraining_jobs.get(task_id, "Task not found")
    return {"task_id": task_id, "status": status}
