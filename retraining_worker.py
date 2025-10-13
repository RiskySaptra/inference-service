import os
import shutil
import zipfile
from train import train_model
from evaluate import evaluate_and_promote

def run_retraining_pipeline(task_id: str, zip_path: str, jobs_store: dict):
    """
    The main worker function for the retraining pipeline.
    """
    jobs_store[task_id] = "starting"
    
    try:
        # Step 1: Unzip the dataset
        jobs_store[task_id] = "unzipping dataset"
        dataset_path = f"datasets/{task_id}"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        # Step 2: Update the dataset.yaml (assuming the new yaml is at the root of the zip)
        jobs_store[task_id] = "configuring dataset"
        shutil.copy(f"{dataset_path}/data.yaml", "dataset.yaml")

        # Step 3: Run the training process
        jobs_store[task_id] = "training model"
        train_model()

        # Step 4: Run the evaluation and promotion process
        jobs_store[task_id] = "evaluating model"
        evaluate_and_promote()

        jobs_store[task_id] = "complete"

    except Exception as e:
        print(f"Error in retraining pipeline (task {task_id}): {e}")
        jobs_store[task_id] = f"failed: {e}"
    
    finally:
        # Clean up the temporary zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)