from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/home/lrbutler/Desktop/611-Yolo-TrafficSigns/data/road_signs/data.yaml",
    epochs=50,
    patience=3,
    imgsz=1024,
    batch=8,
    lr0=0.005,
    augment=True,
    degrees=5,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5
)

# Save trained model
model.export(format="torchscript")
print("Training completed. Model saved.")
