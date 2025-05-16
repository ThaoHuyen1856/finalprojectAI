from ultralytics import YOLO
from pathlib import Path

# === Cấu hình ===
images_dir = Path("khay_com/images")
labels_dir = Path("khay_com/labels")
classes_txt = Path("khay_com/classes.txt")
yaml_output = Path("khay_com/data.yaml")

# === Kiểm tra tồn tại ===
assert images_dir.exists(), " khay_com/images/ không tồn tại"
assert labels_dir.exists(), " khay_com/labels/ không tồn tại"
assert classes_txt.exists(), " khay_com/classes.txt không tồn tại"

# === Đọc tên lớp từ file classes.txt ===
with open(classes_txt, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

# === Sinh đường dẫn tuyệt đối cho YOLO
absolute_data_path = images_dir.parent.resolve().as_posix()  # Đường dẫn thư mục khay_com/

# === Tạo file data.yaml đúng định dạng
yaml_output.write_text(f"""\
path: {absolute_data_path}
train: images
val: images
nc: {len(class_names)}
names: {class_names}
""", encoding="utf-8")

print(f" Đã tạo data.yaml với đường dẫn: {absolute_data_path}")

# === Train YOLOv8
model = YOLO("yolov8m.pt")  # hoặc yolov5s.pt nếu máy yếu

model.train(
    data=str(yaml_output),
    epochs=50,
    imgsz=640,
    batch=4,
    name="khay_detector"
)
