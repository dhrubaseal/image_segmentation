from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='C:\Personal_Projects\projects\Computer Vision\image_segmentation\data\config.yaml', epochs=1, imgsz=640)