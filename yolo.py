from ultralytics import YOLO

# 1. Load a YOLOv8 model (nano variant)
model = YOLO("yolov8n.pt")

# 2. Train, pointing at your data config
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs/train_fathomnet",
    name="yolov8n_fathomnet",
    workers=4,
)

# 3. After training, best weights are at:
print("Best weights:", model.best)
