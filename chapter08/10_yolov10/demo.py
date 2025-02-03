from ultralytics import YOLO
import cv2
import numpy as np

# 加载预训练YOLO模型
model = YOLO("./runs/detect/train/weights/best.pt")

# 读取图像（你可以替换为本地文件路径或使用URL）
img_path = "bus.jpg"  # 这里可以是URL或者本地路径
result = model(img_path)[0]

# 可视化检测结果
# result.show()  # 显示检测结果

# 获取检测框和类别标签
labels = result.names  # 类别标签
confs  = result.boxes.conf.cpu().numpy()
clsids = result.boxes.cls.cpu().numpy()
boxes = result.boxes.xyxy.cpu().numpy()  # 检测框，格式：[x1, y1, x2, y2, confidence, class_id]

# 打印检测结果
for i,box in enumerate(boxes):
    x1, y1, x2, y2 = box
    conf = confs[i]
    class_id = clsids[i]
    label = labels[int(class_id)]
    print(f"Detected {label} with confidence {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# 如果需要对图像进行后续处理或保存
# 将检测框绘制在图像上
img = cv2.imread(img_path) if img_path.endswith(".jpg") else np.array(result.orig_img)
for i,box in enumerate(boxes):
    x1, y1, x2, y2 = box
    conf = confs[i]
    class_id = clsids[i]
    label = labels[int(class_id)]
    color = (0, 255, 0)  # 绿色框
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 保存带框的图像
cv2.imwrite("demo.jpg", img)
