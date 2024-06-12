from ultralytics import YOLO

model = YOLO("Yolo-Weight/yolov8n.pt")

results = model.train(
    data=r"C:\Users\user\Desktop\Semiconductor.v1i.yolov8\data.yaml",
    imgsz=640,
    epochs=200,
    batch=16,
    name="yolov8n_Semiconductor",
)