from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Depends
import uuid
import shutil
import os
from retraining_worker import run_retraining_pipeline
from config import settings
from database import set_job_status, get_job_status
from security import get_api_key

router = APIRouter(dependencies=[Depends(get_api_key)])

@router.post("/retrain/upload/")
async def upload_and_retrain(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a new dataset and start the retraining pipeline in the background.
    """
    task_id = str(uuid.uuid4())
    
    # Save the uploaded zip file
    zip_path = os.path.join(settings.TEMP_ZIP_PATH, f"{task_id}.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Start the background task
    background_tasks.add_task(run_retraining_pipeline, task_id, zip_path)
    set_job_status(task_id, "Retraining started")

    return {"task_id": task_id, "status": "Retraining started"}

@router.get("/retrain/status/{task_id}")
def get_retraining_status(task_id: str):
    """
    Get the status of a retraining job.
    """
    status = get_job_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": status}