from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("runs/detect/train5/weights/last.pt")
    model.train(data="coco128.yaml", epochs=100, imgsz=640, pretrained=False, lr0 = 0.1)