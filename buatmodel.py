import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# Konfigurasi
dataset_folder = "wajah"
img_size = (100, 100)
n_components = 100  # jumlah komponen PCA
model_pca_path = "pca_model.pkl"
model_clf_path = "face_classifier.pkl"

# Load data dari folder per orang
def load_images_and_labels(folder):
    X, y = [], []
    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        if not os.path.isdir(person_folder):
            continue
        for filename in os.listdir(person_folder):
            filepath = os.path.join(person_folder, filename)
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)
                    X.append(img_resized.flatten())
                    y.append(person_name)
    return np.array(X), np.array(y)

# Muat data
print("📥 Memuat gambar dari folder...")
X, y = load_images_and_labels(dataset_folder)
print(f"✅ Total gambar: {len(X)} | Orang dikenali: {len(np.unique(y))}")

# Encode label (nama orang → angka)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# PCA + SVM
print("🧠 Melatih model PCA + SVM...")
pca = PCA(n_components=n_components, whiten=True, random_state=42)
clf = SVC(kernel='linear', probability=True)
model = make_pipeline(pca, clf)

# Training model
model.fit(X, y_encoded)

# Simpan model PCA + SVM
joblib.dump(model, model_clf_path)
joblib.dump(label_encoder, "label_encoder.pkl")
print(f"✅ Model disimpan sebagai: {model_clf_path}")
print(f"✅ Label encoder disimpan sebagai: label_encoder.pkl")
