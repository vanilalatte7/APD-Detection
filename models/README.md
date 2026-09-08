# Pretrained AI Model Weights

Folder ini menyimpan bobot model AI yang digunakan dalam arsitektur **Two-Stage Detection & Classification Pipeline**.

---

## 🧠 Spesifikasi Model

### 1. Deteksi Objek Tahap Pertama (Object Detection)
- **File**: `models/detection/yolov8_apd_best.pt`
- **Arsitektur**: YOLOv8 (Ultralytics)
- **Ukuran**: ~44.8 MB
- **Target Kelas**: Pekerja, Helm, Sepatu
- **Ambang Batas Rekomendasi (`conf`)**: `0.50`
- **Fungsi**: Mendeteksi keberadaan tubuh pekerja serta area helm dan sepatu dalam setiap frame video. Koordinat bounding box kemudian dikirimkan ke model tahap kedua.

---

### 2. Klasifikasi Spesifik Tahap Kedua (Fine-grained Verification)
Area bounding box hasil deteksi dipotong (*cropped*) lalu diverifikasi apakah memenuhi standar APD/K3:

#### a. Klasifikasi Helm (`models/classification/klasifikasi_helm.pt`)
- **Arsitektur**: CNN Classifier / YOLOv8-Cls
- **Ukuran**: ~72.5 MB
- **Kelas Output**: `helm_safety`, `notsafety_helm` / `tanpa_helm`
- **Threshold**: `0.60`

#### b. Klasifikasi Sepatu (`models/classification/klasifikasi_sepatu.pt`)
- **Arsitektur**: CNN Classifier / YOLOv8-Cls (Transfer Learning EfficientNet)
- **Ukuran**: ~72.5 MB
- **Kelas Output**: `sepatu_safety`, `notsafety_sepatu` / `tanpa_sepatu`
- **Threshold**: `0.60`

---

## 💡 Cara Penggunaan dalam Kode Python

```python
from ultralytics import YOLO

# Load model deteksi
model_det = YOLO("models/detection/yolov8_apd_best.pt")

# Load model klasifikasi
model_cls_helm = YOLO("models/classification/klasifikasi_helm.pt")
model_cls_sepatu = YOLO("models/classification/klasifikasi_sepatu.pt")

# Tahap 1: Deteksi
results = model_det("sample_worker.jpg", conf=0.5)

# Tahap 2: Klasifikasi potongan gambar (crop)
# res = model_cls_helm(crop_image)
```
