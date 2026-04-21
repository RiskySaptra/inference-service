from fastapi import APIRouter, File, UploadFile, Query, Form, HTTPException, Depends
from fastapi.security import APIKeyHeader
from PIL import Image
from typing import List, Optional
from pydantic import BaseModel
import io
import uuid
import os
import torch
from ultralytics import YOLO
from config import settings
from security import get_api_key

router = APIRouter(
    dependencies=[Depends(get_api_key)],
    tags=["predict"],
)

model = None

class Prediction(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str
    class_id: int
    detection_id: str

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    count: int

class ErrorResponse(BaseModel):
    message: str

def load_model():
    global model
    try:
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("Running on CPU")
        print(f"Configured device: {settings.DEVICE}")

        if settings.DEVICE == "cpu":
            _orig_load = torch.load
            torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})
            model = YOLO(settings.MODEL_PATH)
            torch.load = _orig_load
        else:
            model = YOLO(settings.MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@router.post(
    "/predict/",
    response_model=PredictResponse,
    summary="Detect objects in an image",
    description="Upload an image and get YOLOv8 object detections with bounding boxes, class labels, and confidence scores.",
    responses={503: {"model": ErrorResponse}},
)
async def predict(
    file: UploadFile = File(..., description="Image file to process"),
    confidence: float = Form(0.45, ge=0.0, le=1.0, description="Confidence threshold"),
    overlap: float = Form(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS"),
    imgsz: int = Form(640, ge=320, le=1280, description="Inference image size"),
    max_det: int = Form(50, ge=1, le=200, description="Maximum detections per image"),
    augment: bool = Form(True, description="Enable test-time augmentation"),
    agnostic_nms: bool = Form(True, description="Enable class-agnostic NMS"),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    image_path = os.path.join(settings.INFERENCE_IMAGES_PATH, f"{uuid.uuid4()}.png")
    image.save(image_path)

    results = model(image, conf=confidence, iou=overlap, imgsz=imgsz, max_det=max_det, augment=augment, agnostic_nms=agnostic_nms, device=settings.DEVICE)

    predictions = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            xywh = box.xywh[0]
            cls_id = int(box.cls[0])
            predictions.append({
                "x": float(xywh[0]),
                "y": float(xywh[1]),
                "width": float(xywh[2]),
                "height": float(xywh[3]),
                "confidence": float(box.conf[0]),
                "class_name": model.names.get(cls_id, str(cls_id)),
                "class_id": cls_id,
                "detection_id": str(uuid.uuid4())
            })

    return {"predictions": predictions}
