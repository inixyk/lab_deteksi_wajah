import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Konfigurasi
dataset_folder = "wajah"
img_size = (100, 100)
batch_size = 16
epochs = 20
model_path = "face_cnn_model.h5"

# Data Augmentasi + Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale',
    shuffle=True
)

num_classes = train_gen.num_classes

# Arsitektur CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Simpan model terbaik selama training
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Latih model
print("🧠 Melatih model CNN...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint]
)

import json

# Ambil label dari train_gen
label_map = train_gen.class_indices
# Balikkan index → label
inverse_map = {v: k for k, v in label_map.items()}

# Simpan ke file JSON
with open("label_map.json", "w") as f:
    json.dump(inverse_map, f)

print("✅ label_map.json berhasil dibuat.")

print(f"✅ Model CNN wajah disimpan ke: {model_path}")
