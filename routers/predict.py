from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Depends
from PIL import Image
import io
import uuid
import os
from ultralytics import YOLO
from config import settings
from security import get_api_key

router = APIRouter(dependencies=[Depends(get_api_key)])

model = None

def load_model():
    global model
    try:
        model = YOLO(settings.MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@router.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0),
    overlap: float = Query(0.5, ge=0.0, le=1.0)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    image_path = os.path.join(settings.INFERENCE_IMAGES_PATH, f"{uuid.uuid4()}.png")
    image.save(image_path)

    results = model(image, conf=confidence, iou=overlap, device=settings.DEVICE)

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
