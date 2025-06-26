import os
import shutil
from sklearn.datasets import fetch_lfw_people
from PIL import Image
import numpy as np

# Konfigurasi
output_folder = "wajah"
min_faces_per_person = 20  # ambil orang yang memiliki >=20 foto wajah
resize_factor = 0.5        # perkecil ukuran gambar

# Ambil dataset LFW
print("📥 Mengunduh dataset LFW...")
lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize_factor, color=True)

# Pastikan folder output ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Simpan sebagian gambar ke folder
print(f"💾 Menyimpan gambar ke folder '{output_folder}'...")
for i, image_array in enumerate(lfw_people.images):
    # Ambil nama orang
    person_name = lfw_people.target_names[lfw_people.target[i]]
    # Buat folder per orang (opsional)
    person_folder = os.path.join(output_folder, person_name.replace(" ", "_"))
    os.makedirs(person_folder, exist_ok=True)

    # Simpan gambar sebagai file jpg
    img = Image.fromarray((image_array * 255).astype(np.uint8))  # Convert to uint8
    filename = os.path.join(person_folder, f"{i}.jpg")
    img.save(filename)

print("✅ Selesai. Gambar disimpan ke folder 'wajah'")
