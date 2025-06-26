import cv2
import numpy as np
import joblib

# Konfigurasi
model_path = "face_classifier.pkl"        # Pipeline PCA + SVM
label_encoder_path = "label_encoder.pkl"  # Label encoder
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
img_size = (100, 100)

# Load model dan label
print("📦 Memuat model dan label encoder...")
model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

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
        face_resized = cv2.resize(face, img_size).flatten().reshape(1, -1)

        # Prediksi nama
        pred = model.predict(face_resized)[0]
        prob = model.predict_proba(face_resized).max()
        name = label_encoder.inverse_transform([pred])[0]

        # Gambar kotak dan nama
        label_text = f"{name} ({prob*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("Face Recognition", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
