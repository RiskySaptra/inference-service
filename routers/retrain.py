from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Depends
import uuid
import shutil
import tempfile
from retraining_worker import run_retraining_pipeline
from database import set_job_status, get_job_status
from security import get_api_key

router = APIRouter()

@router.post("/retrain/upload/", dependencies=[Depends(get_api_key)])
async def upload_and_retrain(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a new dataset and start the retraining pipeline in the background.
    """
    task_id = str(uuid.uuid4())
    
    # Save the uploaded zip file to a temporary file
    try:
        # Create a temporary file that is not deleted on close
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            zip_path = tmp_file.name
    finally:
        file.file.close()

    # Start the background task
    background_tasks.add_task(run_retraining_pipeline, task_id, zip_path)
    set_job_status(task_id, "Retraining started", progress="0%")

    return {"task_id": task_id, "status": "Retraining started"}

@router.get("/retrain/status/{task_id}")
def get_retraining_status(task_id: str):
    """
    Get the status of a retraining job.
    """
    job_info = get_job_status(task_id)
    if job_info is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": job_info.get("status"), "progress": job_info.get("progress")}