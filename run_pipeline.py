from train import train_model
from evaluate import evaluate_and_promote
import os

def run_retraining_pipeline():
    """
    Orchestrates the entire retraining pipeline.
    """
    print("Starting the retraining pipeline...")

    # Step 1: Train the model
    print("\n--- Training Step ---")
    train_model()

    # Step 2: Evaluate and promote the new model
    print("\n--- Evaluation Step ---")
    evaluate_and_promote()

    print("\nPipeline finished.")
    print("If a new model was promoted, please restart the API server to deploy it.")

if __name__ == '__main__':
    # Create the training_runs directory if it doesn't exist
    if not os.path.exists('training_runs'):
        os.makedirs('training_runs')
        
    run_retraining_pipeline()