import os
import cv2
import shutil
import random
from pathlib import Path
from collections import defaultdict

# === Cấu hình ===
IMG_DIR = "khay_com/images"
LABEL_DIR = "khay_com/labels"
CLASSES_FILE = "khay_com/classes.txt"
OUTPUT_DIR = "cnn_dataset"
MAX_PER_CLASS = 80
MIN_PER_CLASS = 30

# === Đọc danh sách nhãn
with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]

os.makedirs(OUTPUT_DIR, exist_ok=True)
for cls in class_names:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# === Crop ảnh từ YOLO segment (polygon)
class_counts = defaultdict(int)

for label_file in Path(LABEL_DIR).glob("*.txt"):
    image_file = Path(IMG_DIR) / (label_file.stem + ".jpg")
    if not image_file.exists():
        continue

    img = cv2.imread(str(image_file))
    h, w = img.shape[:2]

    with open(label_file, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 9:
                continue  # Bỏ qua nếu không đủ điểm polygon

            cls_id = int(parts[0])
            coords = list(map(float, parts[1:9]))
            points = [(coords[j]*w, coords[j+1]*h) for j in range(0, len(coords), 2)]

            xs, ys = zip(*points)
            x1, y1 = int(max(min(xs), 0)), int(max(min(ys), 0))
            x2, y2 = int(min(max(xs), w-1)), int(min(max(ys), h-1))

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue  # Bỏ crop nhỏ

            crop = img[y1:y2, x1:x2]
            label = class_names[cls_id]
            save_path = os.path.join(OUTPUT_DIR, label, f"{label}_{label_file.stem}_{i}.jpg")

            if class_counts[label] < MAX_PER_CLASS:
                cv2.imwrite(save_path, crop)
                class_counts[label] += 1

print(f" Đã crop xong ảnh. Số lượng mỗi lớp:")
for cls, count in class_counts.items():
    print(f"- {cls}: {count}")

# === Hàm tăng cường ảnh đơn giản
def augment_image(img):
    ops = [
        lambda i: cv2.flip(i, 1),  # lật ngang
        lambda i: cv2.convertScaleAbs(i, alpha=1.2, beta=10),  # tăng sáng
        lambda i: cv2.GaussianBlur(i, (3, 3), 0)  # làm mờ nhẹ
    ]
    op = random.choice(ops)
    return op(img)


# === Augment lớp hiếm
print("\n Tăng cường dữ liệu cho lớp < 30 ảnh:")
for cls in class_names:
    cls_dir = os.path.join(OUTPUT_DIR, cls)
    existing = list(Path(cls_dir).glob("*.jpg"))

    if not existing:
        print(f" Bỏ qua {cls} vì không có ảnh.")
        continue

    while len(existing) < MIN_PER_CLASS:
        while True:
            src_path = str(random.choice(existing))
            src = cv2.imread(src_path)
            if src is not None:
                break
            else:
                print(f" Không đọc được ảnh: {src_path}, thử lại...")

        aug = augment_image(src)
        fname = f"aug_{len(existing)}.jpg"
        save_path = os.path.join(cls_dir, fname)
        cv2.imwrite(save_path, aug)
        existing.append(save_path)

    print(f"- {cls}: {len(existing)} ảnh sau tăng cường")

print("\n Đã tạo xong tập dữ liệu CNN tại thư mục:", OUTPUT_DIR)
