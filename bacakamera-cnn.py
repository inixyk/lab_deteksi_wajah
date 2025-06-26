import cv2
import numpy as np
import tensorflow as tf
import json
import os

# Konfigurasi
model_path = 'face_cnn_model.h5'
label_path = 'label_map.json'
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
img_size = (100, 100)

# Load model CNN
print("📦 Memuat model CNN...")
model = tf.keras.models.load_model(model_path)

# Load label map (index → nama)
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}
else:
    print("❌ label_map.json tidak ditemukan.")
    exit()

# Load face detector
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Tidak dapat membuka kamera.")
    exit()

print("📷 Kamera aktif. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, img_size)
        face_normalized = face_resized.astype('float32') / 255.0
        face_input = np.expand_dims(face_normalized, axis=(0, -1))  # shape: (1, 100, 100, 1)

        # Prediksi wajah
        predictions = model.predict(face_input)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        name = label_map.get(predicted_class, "Tidak dikenal")
        label_text = f"{name} ({confidence*100:.1f}%)"

        # Tampilkan kotak dan nama
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    # Tampilkan hasil
    cv2.imshow("Face Recognition with CNN", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
