from ultralytics import YOLO
import os
import torch
from database import set_job_status

def train_model(task_id: str):
    """
    This function trains a YOLOv8 model on a custom dataset.
    """
    # Determine the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Path to the dataset configuration file
    data_config_path = 'dataset.yaml'

    # Path to the current best model
    # The pipeline will use the latest model from the 'models' directory
    model_path = 'models/v1.pt'  # IMPORTANT: Update this to your current best model

    # Check if a model exists to fine-tune from
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Starting with a pre-trained YOLOv8 model.")
        model_path = 'yolov8n.pt'  # Fallback to a standard pre-trained model

    # Load the model
    model = YOLO(model_path)

    def on_epoch_end(trainer):
        epoch = trainer.epoch
        total_epochs = trainer.epochs
        # Only update status every 5 epochs to reduce DB writes, and on the last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == total_epochs:
            progress = (epoch + 1) / total_epochs * 100
            set_job_status(task_id, "training model", progress=f"{progress:.2f}%")

    model.add_callback("on_epoch_end", on_epoch_end)

    # Train the model
    # The results are saved to a 'runs' directory by default
    results = model.train(
        data=data_config_path,
        epochs=100,  # Adjust the number of epochs as needed
        imgsz=640,
        project='training_runs',
        name=task_id,
        device=device
    )

    print("Training complete.")
    print(f"The new model and training results are saved in the 'training_runs/latest_run' directory.")

if __name__ == '__main__':
    train_model("task_id_from_cli")