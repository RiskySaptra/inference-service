from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Depends
from PIL import Image
import io
import uuid
import os
from ultralytics import YOLO
from config import settings
from security import get_api_key

router = APIRouter(dependencies=[Depends(get_api_key)])

# Load the YOLOv8 model
try:
    model = YOLO(settings.MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@router.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0),
    overlap: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Endpoint to receive an image and return YOLOv8 predictions.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Save the image
    image_path = os.path.join(settings.INFERENCE_IMAGES_PATH, f"{uuid.uuid4()}.png")
    image.save(image_path)

    # Perform inference
    results = model(image, conf=confidence, iou=overlap, device=settings.DEVICE)

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