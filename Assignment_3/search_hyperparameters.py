from ultralytics import YOLO
import ray
from ray import tune
import os

def trainable(config):
    model = YOLO("yolov8n.pt")

    model.train(
        data="/home/lrbutler/Desktop/611-Yolo-TrafficSigns/road_signs/data.yaml",
        epochs=50,
        patience=3,
        imgsz=config["imgsz"],
        batch=config["batch"],
        lr0=config["lr0"],
        augment=True,
        degrees=5,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5
    )

    # Save trained model
    model.export(format="torchscript")
    print("Training completed. Model saved.")


if __name__ == "__main__":

    # Define Hyperparameter Search Space
    search_space = {
        "lr0": tune.loguniform(1e-4, 1e-2),  # Learning rate
        "batch": tune.choice([8, 16, 32]),  # Batch sizes
        "imgsz": tune.choice([640, 1024, 1280]),  # Image sizes
    }

    ray.init(ignore_reinit_error=True)

    resources={"cpu": 1, "gpu": 1}

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=18,
            max_concurrent_trials=2
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.join(os.getcwd(), 'training_results')
        )
    )
    tuner.fit()
