import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# === Cấu hình ===
IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 30
DATASET_DIR = "cnn_dataset"

# === Chuẩn hóa và tăng cường dữ liệu ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1]
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# === Ghi lại nhãn vào labels.pkl ===
with open("labels.pkl", "wb") as f:
    pickle.dump(train_gen.class_indices, f)

# === Mô hình CNN đơn giản ===
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# === Huấn luyện ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# === Lưu mô hình dưới dạng HDF5 (.h5)
model.save("cnn.h5")

# === Vẽ biểu đồ huấn luyện
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
