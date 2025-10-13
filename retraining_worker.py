import os
import shutil
import zipfile
import yaml
from train import train_model
from evaluate import evaluate_and_promote
from database import set_job_status
from config import settings

def _unzip_dataset(task_id: str, zip_path: str) -> str:
    """Unzips the dataset and returns the path to the extracted files."""
    set_job_status(task_id, "unzipping dataset", progress="10%")
    dataset_path = os.path.join(settings.DATASETS_PATH, task_id)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    set_job_status(task_id, "dataset unzipped", progress="20%")
    return dataset_path

def _configure_dataset_yaml(task_id: str, dataset_path: str):
    """Configures the dataset.yaml file for training."""
    set_job_status(task_id, "configuring dataset", progress="25%")
    
    # Read the original data.yaml
    with open(f"{dataset_path}/data.yaml", 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    # Set the path to the dataset directory
    data_yaml['path'] = dataset_path
    
    # Clean up relative paths
    for key in ['train', 'val', 'test']:
        if key in data_yaml and isinstance(data_yaml[key], str):
            clean_path = data_yaml[key]
            while clean_path.startswith('../'):
                clean_path = clean_path[3:]
            data_yaml[key] = clean_path

    # Write the updated data to the root dataset.yaml
    with open("dataset.yaml", 'w') as f:
        yaml.dump(data_yaml, f)
    set_job_status(task_id, "dataset configured", progress="30%")

def run_retraining_pipeline(task_id: str, zip_path: str):
    """
    The main worker function for the retraining pipeline.
    """
    set_job_status(task_id, "starting", progress="0%")
    
    try:
        dataset_path = _unzip_dataset(task_id, zip_path)
        _configure_dataset_yaml(task_id, dataset_path)

def _run_training(task_id: str):
    """Runs the model training process."""
    set_job_status(task_id, "training model", progress="30%")
    train_model(task_id)
    set_job_status(task_id, "model training complete", progress="90%")

def _evaluate_and_promote_model(task_id: str):
    """Evaluates the new model and promotes it if it's better."""
    set_job_status(task_id, "evaluating model", progress="90%")
    evaluate_and_promote(task_id)
    set_job_status(task_id, "model evaluation complete", progress="95%")

def run_retraining_pipeline(task_id: str, zip_path: str):
    """
    The main worker function for the retraining pipeline.
    """
    set_job_status(task_id, "starting", progress="0%")
    
    try:
        dataset_path = _unzip_dataset(task_id, zip_path)
        _configure_dataset_yaml(task_id, dataset_path)
        _run_training(task_id)
        _evaluate_and_promote_model(task_id)

        set_job_status(task_id, "complete", progress="100%")

    except Exception as e:
        print(f"Error in retraining pipeline (task {task_id}): {e}")
        set_job_status(task_id, f"failed: {e}", progress="-1%")
    
    finally:
        # Clean up the temporary zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)