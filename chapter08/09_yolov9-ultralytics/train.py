# 使用说明：https://docs.ultralytics.com/zh/usage/python/
# 预训练模型下载：https://github.com/ultralytics/assets/releases
# 预训练模型下载：https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9s.pt
# ultralytics-8.3.39

from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov9s.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov9s.pt")

# Train the model using the 'coco8.yaml' dataset for 300 epochs
results = model.train(data="data/coco128.yaml", epochs=300, batch=8, imgsz=640)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
# success = model.export(format="onnx")