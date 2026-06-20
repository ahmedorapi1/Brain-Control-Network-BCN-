from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data=r"D:\GP_data\vision_data\data.yaml",
    epochs=100,
    imgsz=960,
    batch=4,
    device="cpu",
    lr0=0.001,
    optimizer="auto"
)
