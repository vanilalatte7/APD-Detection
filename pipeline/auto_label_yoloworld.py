import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO 

# ==========================================
# 1. PENGATURAN UTAMA
# ==========================================
video_path = r"D:\Fari Hadi P\yusha\Pengolahan Dataset\lv_0_20260508223139.mp4"
model_path = r"D:\Fari Hadi P\yusha\Pengolahan Dataset\yolov8s-world.pt"
base_dir = "Dataset_AH"
img_dir = os.path.join(base_dir, "images")
lbl_dir = os.path.join(base_dir, "labels")

TARGET_COUNT = 600  # Dinaikkan sedikit agar lebih banyak ruang untuk sepatu
CONF_THRESHOLD = 0.3 # Fokus pada benda kecil (sepatu)
MIN_GAP_SECONDS = 1  # Jeda lebih rapat
STRIDE = 30           # Scan setiap 1 detik (Thorough Mode)

def calculate_sharpness(image):
    """Menghitung tingkat ketajaman gambar menggunakan Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ==========================================
# 2. PERSIAPAN FOLDER
# ==========================================
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

# ==========================================
# 3. INISIALISASI MODEL & VIDEO
# ==========================================
print("⏳ Memuat model AI...")
# Gunakan GPU jika tersedia
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Menggunakan device: {device}")

model = YOLO(model_path).to(device) 

# KEAJAIBAN YOLO-WORLD: Tentukan objek apa saja yang ingin dicari
print("🎯 Mengatur target deteksi: Person, Helmet, Shoe...")
model.set_classes(["person", "helmet", "shoe"])

print("🏷️ Daftar Kelas Baru:")
for id, name in model.names.items():
    print(f"   - ID {id}: {name}")

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"🎬 Video: {total_frames} frame (~{total_frames/(fps*60):.1f} menit).")
print(f"⏩ Mode Teliti: STRIDE = {STRIDE} (Cek setiap detik).")

# ==========================================
# 4. PASS 1: SCANNING & SCORING
# ==========================================
print(f"\n🔍 PASS 1: Mencari kandidat (Fokus: Sepatu/Shoes)...")
candidates = []

frame_idx = 0
while cap.isOpened():
    if frame_idx % STRIDE != 0:
        ret = cap.grab()
        if not ret: break
        frame_idx += 1
        continue
        
    ret, frame = cap.read()
    if not ret:
        break
    
    current_frame = frame_idx
    frame_idx += 1
    
    # Prediksi
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    detected_classes = [int(box.cls[0]) for box in boxes]
    unique_classes = set(detected_classes)

    if len(unique_classes) >= 1:
        avg_conf = np.mean([float(box.conf[0]) for box in boxes])
        sharpness = calculate_sharpness(frame)
        
        # Skor dasar
        base_score = avg_conf * sharpness
        
        # BONUS SPESIFIK: Berikan booster jika ada SEPATU (ID 2)
        if 2 in unique_classes:
            base_score *= 5  # Lipat gandakan skor jika ada sepatu
        
        candidates.append({
            'frame_idx': current_frame,
            'base_score': base_score,
            'classes': detected_classes,
            'unique_count': len(unique_classes),
            'boxes': [(int(box.cls[0]), box.xywhn[0].tolist()) for box in boxes]
        })
        
        if len(candidates) % 100 == 0:
            print(f"📍 Menemukan {len(candidates)} kandidat (Termasuk sepatu)...")

cap.release()

# ==========================================
# 5. PASS 2: SELEKSI TERBAIK, UNIK & SEIMBANG
# ==========================================
print(f"\n🎯 PASS 2: Menyeleksi {TARGET_COUNT} frame (Prioritas: Keragaman Kelas & Kualitas)...")

selected_frames = []
class_distribution = {} # Menghitung jumlah instance per kelas
min_frame_gap = MIN_GAP_SECONDS * fps

# Urutkan awal berdasarkan jumlah kelas unik terbanyak, lalu skor kualitas
candidates.sort(key=lambda x: (x['unique_count'], x['base_score']), reverse=True)

while len(selected_frames) < TARGET_COUNT and candidates:
    # Strategi Seleksi Seimbang:
    # Berikan bonus pada frame yang mengandung kelas yang masih sedikit jumlahnya di dataset
    best_candidate_idx = -1
    max_final_score = -1
    
    # Untuk efisiensi, kita hanya cek 200 kandidat teratas di setiap iterasi
    search_limit = min(200, len(candidates))
    
    for i in range(search_limit):
        cand = candidates[i]
        
        # Cek jeda waktu (Temporal Uniqueness)
        if any(abs(cand['frame_idx'] - s['frame_idx']) < min_frame_gap for s in selected_frames):
            continue
            
        # Hitung skor keseimbangan kelas
        # Semakin jarang kelas tersebut sudah diambil, semakin tinggi bonusnya
        balance_bonus = 0
        for cls_id in set(cand['classes']):
            count = class_distribution.get(cls_id, 0)
            balance_bonus += (1.0 / (count + 1))
            
        # Skor akhir = (Kualitas) * (Jumlah Kelas Unik) * (Bonus Keseimbangan)
        final_score = cand['base_score'] * cand['unique_count'] * (1 + balance_bonus)
        
        if final_score > max_final_score:
            max_final_score = final_score
            best_candidate_idx = i
            
    if best_candidate_idx != -1:
        chosen = candidates.pop(best_candidate_idx)
        selected_frames.append(chosen)
        
        # Update statistik distribusi kelas
        for cls_id in chosen['classes']:
            class_distribution[cls_id] = class_distribution.get(cls_id, 0) + 1
    else:
        # Jika tidak ada yang memenuhi syarat gap di search_limit, hapus beberapa agar tidak stuck
        candidates.pop(0)

print(f"✅ Berhasil memilih {len(selected_frames)} frame seimbang.")
print("📊 Distribusi Kelas:")
for cls_id, count in sorted(class_distribution.items()):
    print(f"   - Kelas {cls_id}: {count} objek")

# ==========================================
# 6. PASS 3: EKSTRAKSI & PENYIMPANAN
# ==========================================
print(f"\n💾 PASS 3: Mengekstrak dan menyimpan hasil...")
cap = cv2.VideoCapture(video_path)
saved_count = 0

selected_frames.sort(key=lambda x: x['frame_idx'])

for item in selected_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, item['frame_idx'])
    ret, frame = cap.read()
    if not ret:
        continue
    
    file_name = f"dataset_best_{saved_count:04d}_f{item['frame_idx']}"
    
    cv2.imwrite(os.path.join(img_dir, f"{file_name}.jpg"), frame)
    
    with open(os.path.join(lbl_dir, f"{file_name}.txt"), "w") as f:
        for cls_id, (x_c, y_c, w, h) in item['boxes']:
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    
    saved_count += 1
    if saved_count % 50 == 0:
        print(f"📥 Tersimpan {saved_count}/{len(selected_frames)}...")

cap.release()
print(f"\n🎉 PROSES SELESAI! Total dataset: {saved_count} gambar.")
print(f"📁 Lokasi: {os.path.abspath(base_dir)}")

