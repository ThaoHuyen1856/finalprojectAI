from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# === Load model YOLO và CNN ===
yolo_model = YOLO("runs/detect/khay_detector8/weights/best.pt")
cnn_model = load_model("cnn.h5")

with open("labels.pkl", "rb") as f:
    class_indices = pickle.load(f)
food_labels = [label for label, idx in sorted(class_indices.items(), key=lambda x: x[1])]

def detect_and_crop_food(img, padding=60, yolo_conf_threshold=0.6, nms_iou_threshold=0.45):
    height, width = img.shape[:2]
    results = yolo_model(img, conf=yolo_conf_threshold, iou=nms_iou_threshold)
    crops = []

    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, class_ids):
        if conf >= yolo_conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            crop = img[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            yolo_label = food_labels[cls_id]
            crops.append((crop, conf, yolo_label))
    return crops

def predict_food(image_crop, cnn_model, food_labels, cnn_conf_threshold=0.75):
    img = cv2.resize(image_crop, (200, 200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    pred = cnn_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
    idx = np.argmax(pred)
    conf = pred[idx]
    label = food_labels[idx]

    if conf >= cnn_conf_threshold:
        return label, conf
    else:
        return "Unknown", conf
