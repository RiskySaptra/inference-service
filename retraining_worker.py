import os
import shutil
import zipfile
import yaml
from train import train_model
from evaluate import evaluate_and_promote
from database import set_job_status

def run_retraining_pipeline(task_id: str, zip_path: str):
    """
    The main worker function for the retraining pipeline.
    """
    set_job_status(task_id, "starting")
    
    try:
        # Step 1: Unzip the dataset
        set_job_status(task_id, "unzipping dataset")
        dataset_path = f"datasets/{task_id}"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        # Step 2: Update the dataset.yaml (assuming the new yaml is at the root of the zip)
        set_job_status(task_id, "configuring dataset")
        
        # Read the original data.yaml
        with open(f"{dataset_path}/data.yaml", 'r') as f:
            data_yaml = yaml.safe_load(f)
        
        # Set the path to the dataset directory. Ultralytics will resolve other paths relative to this.
        data_yaml['path'] = dataset_path
        
        # Clean up any problematic relative paths (e.g., '../')
        for key in ['train', 'val', 'test']:
            if key in data_yaml and isinstance(data_yaml[key], str):
                clean_path = data_yaml[key]
                while clean_path.startswith('../'):
                    clean_path = clean_path[3:]
                data_yaml[key] = clean_path

        # Write the updated data to the root dataset.yaml
        with open("dataset.yaml", 'w') as f:
            yaml.dump(data_yaml, f)

        # Step 3: Run the training process
        set_job_status(task_id, "training model")
        train_model()

        # Step 4: Run the evaluation and promotion process
        set_job_status(task_id, "evaluating model")
        evaluate_and_promote()

        set_job_status(task_id, "complete")

    except Exception as e:
        print(f"Error in retraining pipeline (task {task_id}): {e}")
        set_job_status(task_id, f"failed: {e}")
    
    finally:
        # Clean up the temporary zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)