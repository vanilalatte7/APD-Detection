import cv2
import os
from pathlib import Path

# ==========================================
# 1. PENGATURAN PATH
# ==========================================
dataset_dir = "datasetAIAH"
img_dir = os.path.join(dataset_dir, "images")
lbl_dir = os.path.join(dataset_dir, "labels")
output_dir = "Cropped_Helmets"

# ID Class untuk HELM (berdasarkan label YOLO sebelumnya)
HELM_CLASS_ID = "0"

# Buat folder output jika belum ada
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📁 Folder output dibuat: {output_dir}")

# ==========================================
# 2. PROSES CROP
# ==========================================
label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
crop_count = 0

print(f"🚀 Memulai proses cropping helm dari {len(label_files)} file label...")

for lbl_file in label_files:
    # 1. Baca isi label
    with open(os.path.join(lbl_dir, lbl_file), 'r') as f:
        lines = f.readlines()

    # 2. Cari baris yang berisi helm (Class ID 1)
    helm_detections = [line.strip().split() for line in lines if line.startswith(HELM_CLASS_ID)]

    if not helm_detections:
        continue

    # 3. Cari gambar yang sesuai
    img_name_base = Path(lbl_file).stem
    img_file = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        temp_path = os.path.join(img_dir, img_name_base + ext)
        if os.path.exists(temp_path):
            img_file = temp_path
            break

    if img_file is None:
        continue

    # 4. Load gambar
    image = cv2.imread(img_file)
    if image is None:
        continue
    h, w, _ = image.shape

    # 5. Lakukan Crop untuk setiap helm yang ditemukan
    for i, det in enumerate(helm_detections):
        _, x_center, y_center, width, height = map(float, det)

        # Konversi ke koordinat pixel
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # Pastikan koordinat di dalam batas gambar
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Crop gambar
        helm_crop = image[y1:y2, x1:x2]

        if helm_crop.size == 0:
            continue

        # Simpan hasil crop
        save_name = f"helm_{img_name_base}_{i}.jpg"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, helm_crop)
        crop_count += 1

print(f"\n✅ SELESAI!")
print(f"⛑️ Total helm yang berhasil di-crop: {crop_count}")
print(f"📂 Hasil tersimpan di: {os.path.abspath(output_dir)}")
