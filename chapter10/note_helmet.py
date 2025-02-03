https://github.com/ultralytics/yolov5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt


cp data/coco128.yaml data/helmet.yaml
path: ../data/helmet_data # dataset root dir
train: train # train dir
val: val # val dir
test: test # test dir
# Classes
names:
  0: helmet
  1: head


parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
parser.add_argument("--data", type=str, default=ROOT / "data/helmet.yaml", help="dataset.yaml path")
parser.add_argument("--epochs", type=int, default=300, help="total training epochs")


python export.py --include onnx
python detect.py --weights runs/train/exp5/weights/best.onnx









