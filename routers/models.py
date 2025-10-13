from fastapi import APIRouter, HTTPException, Depends
import os
from config import settings
from security import get_api_key
from pydantic import BaseModel
from routers import predict

router = APIRouter(dependencies=[Depends(get_api_key)])

ACTIVE_MODEL_FILE = "active_model.txt"

class Model(BaseModel):
    task_id: str

def get_active_model():
    """Gets the path of the active model."""
    if not os.path.exists(ACTIVE_MODEL_FILE):
        return settings.MODEL_PATH  # Default model
    with open(ACTIVE_MODEL_FILE, 'r') as f:
        task_id = f.read().strip()
    return f"training_runs/{task_id}/weights/best.pt"

@router.get("/models/")
def list_models():
    """Lists all available trained models."""
    models = []
    if os.path.exists("training_runs"):
        for task_id in os.listdir("training_runs"):
            model_path = f"training_runs/{task_id}/weights/best.pt"
            if os.path.exists(model_path):
                models.append(task_id)
    return {"models": models}

@router.post("/models/set_active/")
def set_active_model(model: Model):
    """Sets the active model for predictions."""
    model_path = f"training_runs/{model.task_id}/weights/best.pt"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    with open(ACTIVE_MODEL_FILE, 'w') as f:
        f.write(model.task_id)
    
    # Reload the model in the predict router
    predict.load_model()
    
    return {"message": f"Model {model.task_id} set as active and reloaded"}