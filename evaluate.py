from ultralytics import YOLO
import os
import shutil

def evaluate_and_promote():
    """
    Evaluates the newly trained model and promotes it if it performs better.
    """
    # Path to the new model (best.pt from the latest training run)
    new_model_path = 'training_runs/latest_run/weights/best.pt'

    # Path to the current production model
    prod_model_path = 'models/v1.pt'  # IMPORTANT: Update this to your current best model

    # Check if a new model was trained
    if not os.path.exists(new_model_path):
        print("No new model found to evaluate.")
        return

    # Load the models
    new_model = YOLO(new_model_path)
    
    # If there's no production model, the new model is promoted by default
    if not os.path.exists(prod_model_path):
        print("No production model found. Promoting the new model.")
        shutil.copy(new_model_path, prod_model_path)
        return

    prod_model = YOLO(prod_model_path)

    # Evaluate both models on the validation set
    print("Evaluating the new model...")
    new_model_metrics = new_model.val(data='dataset.yaml')
    
    print("Evaluating the production model...")
    prod_model_metrics = prod_model.val(data='dataset.yaml')

    # Compare the performance (using mAP50-95)
    new_map = new_model_metrics.box.map
    prod_map = prod_model_metrics.box.map

    print(f"New model mAP: {new_map}")
    print(f"Production model mAP: {prod_map}")

    # Promote the new model if it's better
    if new_map > prod_map:
        print("New model performs better. Promoting to production.")
        shutil.copy(new_model_path, prod_model_path)
    else:
        print("Production model performs better. Keeping the current model.")

if __name__ == '__main__':
    evaluate_and_promote()