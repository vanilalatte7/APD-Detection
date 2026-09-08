# 🦺 Sistem Deteksi Pelanggaran APD (K3) Berbasis Machine Learning & Real-time Monitoring

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Framework-Flask-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MySQL](https://img.shields.io/badge/Database-MySQL-4479A1?logo=mysql&logoColor=white)](https://www.mysql.com/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Hardware](https://img.shields.io/badge/Target%20Hardware-Raspberry%20Pi%204%20%2F%20Edge%20PC-C51A4A?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sistem Cerdas Pengawasan K3 Terintegrasi: Deteksi Otomatis Helm & Sepatu Safety, Pencatatan Bukti Pelanggaran ke Database MySQL, dan Dashboard Monitoring Real-time.**

[Fitur Utama](#-fitur-utama) • [Arsitektur Sistem](#-arsitektur-sistem) • [Dataset & Training](#-dataset--pelatihan-model) • [Panduan Instalasi](#-panduan-instalasi--penggunaan) • [Struktur Proyek](#-struktur-direktori)

</div>

---

## 📌 Latar Belakang

Kepatuhan terhadap penggunaan **Alat Pelindung Diri (APD)** merupakan pilar fundamental dalam keselamatan dan kesehatan kerja (K3) pada sektor industri, manufaktur, konstruksi, dan fasilitas berisiko tinggi. Pengawasan manual secara konvensional memiliki keterbatasan jangkauan, rawan *human error*, serta membutuhkan biaya operasional yang besar.

Proyek ini menghadirkan solusi **Automated K3 Vision Monitoring** menggunakan arsitektur kecerdasan buatan dua tahap (*Two-Stage AI Pipeline*):
1. **Deteksi Objek (Stage 1)**: Menggunakan **YOLOv8** untuk mendeteksi pekerja, area helm, dan area sepatu secara *real-time*.
2. **Klasifikasi Khusus (Stage 2)**: Menggunakan model **CNN (Convolutional Neural Network)** untuk memverifikasi apakah helm dan sepatu yang terdeteksi memenuhi standar safety atau melanggar aturan K3.
3. **Auto-Capture & Database Logging**: Bukti pelanggaran secara otomatis difoto, diberi bounding box berwarna merah, dicatat waktu dan tingkat akurasinya, lalu disimpan ke database **MySQL**.
4. **Interactive Web Dashboard**: Menampilkan live streaming feed, indikator peringatan bahaya, dan tabel riwayat investigasi insiden.

---

## 🏗️ Arsitektur Sistem

Sistem dirancang untuk dapat diimplementasikan pada edge device seperti **Raspberry Pi 4** maupun server lokal yang terhubung ke jaringan CCTV pabrik:

<div align="center">
  <img src="docs/architecture-diagram.png" alt="Arsitektur Sistem APD Detection" width="850"/>
  <p><em>Diagram Arsitektur Fisik & Jaringan Sistem (CCTV → DVR/NVR → Router/Switch → Raspberry Pi 4 / Server → Web Client)</em></p>
</div>

### 🔄 Alur Kerja Sistem (Flowchart)

<div align="center">
  <img src="docs/system-flowchart.png" alt="Flowchart Sistem APD Detection" width="750"/>
  <p><em>Alur Pemrosesan Frame CCTV, Verifikasi Stage-2 CNN, Auto-Capture Bukti Pelanggaran, dan Penyimpanan ke MySQL</em></p>
</div>

---

## ✨ Fitur Utama

- 🎯 **Two-Stage Precision AI Pipeline**: Menggabungkan kecepatan deteksi objek YOLOv8 dengan ketelitian klasifikasi CNN untuk meminimalkan *false alarm*.
- 📹 **Multi-Source Video Input**: Mendukung input kamera USB (Webcam), IP Camera via protokol **RTSP**, dan pemutaran file video rekaman (`.mp4`, `.avi`).
- ⚡ **Auto-Capture Bukti Pelanggaran**: Mengabadikan foto bukti saat terjadi pelanggaran dengan interval cerdas (debounce 7 detik) untuk mencegah penumpukan data duplikat.
- 🗄️ **Pencatatan Database Terstruktur**: Log pelanggaran mencakup waktu kejadian, jenis pelanggaran, detail deskripsi, tingkat keyakinan (confidence), dan file gambar bukti.
- 🖥️ **Web Dashboard Modern & Responsif**: Dibangun dengan Flask & Bootstrap 5, dilengkapi sistem autentikasi pengguna (Login/Register), live stream video monitor, dan tab riwayat pelanggaran.
- 🧪 **Pipeline Dataset Mandiri**: Dilengkapi skrip auto-labeling berbasis **YOLO-World** dan skrip auto-crop untuk mempermudah penambahan data baru.

---

## 📊 Dataset & Pelatihan Model

Proses pembuatan model dilakukan melalui tahapan pipeline otomatis:

1. **Auto-Labeling**: Menggunakan model zero-shot `yolov8s-world.pt` untuk melabeli frame video secara otomatis.
2. **Dataset Annotation & Split**: Verifikasi label dan pembagian dataset (80% Train, 20% Validation) pada Roboflow.
3. **Training YOLOv8 (Stage 1)**: Pelatihan model deteksi menggunakan GPU T4 di Google Colab.
4. **Object Cropping**: Memotong objek helm dan sepatu dari hasil bounding box deteksi untuk dataset klasifikasi.
5. **Transfer Learning CNN (Stage 2)**: Melatih model klasifikasi berbasis **EfficientNetB0** untuk membedakan APD standar vs non-standar.

🔗 **Link Google Colab Notebook:**  
[Google Colab Training Notebook](https://colab.research.google.com/drive/1w37FpfkGKSxxVAAef55YZxGcfffT4Qg3?usp=sharing)

> Detail panduan lengkap proses olah data dapat dilihat di [pipeline/README.md](pipeline/README.md).

---

## 📁 Struktur Direktori

```
APD-Detection/
├── .github/                        # Template Issue & Pull Request GitHub
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
├── database/                       # Skema database MySQL
│   └── schema.sql                  # Script SQL pembuatan database k3monitoring & tabel
├── docs/                           # Dokumentasi & diagram sistem
│   ├── architecture-diagram.png    # Diagram arsitektur fisik perangkat
│   ├── system-flowchart.png        # Diagram alur logika deteksi & peringatan
│   └── architecture.drawio         # File mentah Draw.io
├── models/                         # Bobot model AI pretrained
│   ├── detection/
│   │   └── yolov8_apd_best.pt      # Bobot model deteksi objek YOLOv8
│   ├── classification/
│   │   ├── klasifikasi_helm.pt     # Bobot classifier helm K3
│   │   └── klasifikasi_sepatu.pt   # Bobot classifier safety shoes
│   └── README.md                   # Dokumentasi spesifikasi model
├── pipeline/                       # Skrip pengolahan dataset & training
│   ├── auto_crop_helmets.py        # Ekstraksi crop helm dari bounding box
│   ├── auto_crop_shoes.py          # Ekstraksi crop sepatu dari bounding box
│   ├── auto_label_yoloworld.py     # Script auto-labeling YOLO-World
│   ├── train_cnn_shoes_colab.py    # Training transfer learning EfficientNetB0
│   ├── test_model_inference.py     # Pengujian inferensi video/gambar
│   └── README.md                   # Panduan pipeline dataset
├── scripts/                        # Utilitas & pengujian mandiri
│   └── webcam_detector.py          # Pengujian deteksi langsung via webcam/CCTV
├── web/                            # Web Application Flask K3 Monitoring
│   ├── app.py                      # Server Flask utama (Streaming, DB, & AI Inference)
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css           # Styling dashboard
│   │   └── uploads/                # Direktori penyimpanan bukti foto pelanggaran
│   └── templates/
│       └── index.html              # Template dashboard web terpadu
├── .env.example                    # Template konfigurasi environment
├── .gitignore                      # Mengabaikan file video mentah, zip besar, & venv
├── LICENSE                         # Lisensi MIT
├── README.md                       # Dokumentasi utama proyek
└── requirements.txt                # Daftar dependensi Python
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Prasyarat Sistem
- Python 3.8 - 3.10
- MySQL Server (XAMPP / MySQL Community Server)
- Webcam, IP Camera (RTSP), atau file video rekaman

### 2. Clone Repositori
```bash
git clone https://github.com/vanilalatte7/APD-Detection.git
cd APD-Detection
```

### 3. Buat Virtual Environment & Install Dependensi
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / Raspberry Pi OS
python3 -m venv .venv
source .venv/bin/activate

# Install dependensi
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Setup Database MySQL
Buka MySQL client / phpMyAdmin, lalu import skrip database:
```bash
mysql -u root -p < database/schema.sql
```
*Atau buka `database/schema.sql` dan jalankan isinya pada phpMyAdmin / MySQL Workbench.*

### 5. Konfigurasi Lingkungan (.env)
Salin file template `.env.example` menjadi `.env`:
```bash
cp .env.example .env     # Linux / Mac
copy .env.example .env   # Windows
```
Sesuaikan konfigurasi:
- `VIDEO_SOURCE`: Isi `0` untuk webcam, atau `rtsp://admin:password@192.168.1.64:554/stream` untuk CCTV.
- `DATABASE_URL`: Isi kredensial MySQL Anda (default: `mysql+mysqlconnector://root:@localhost/k3monitoring`).

### 6. Menjalankan Server Web Monitoring
```bash
cd web
python app.py
```
Buka browser Anda dan akses:  
👉 **`http://localhost:5000`**

- **Akun Default Admin:**
  - **Username:** `admin`
  - **Password:** `admin123`

### 7. Menjalankan Deteksi Standalone via Script (Opsional)
Jika ingin menguji deteksi langsung di layar tanpa server web:
```bash
python scripts/webcam_detector.py --source 0 --conf-det 0.5
```

---

## 🛡️ Lisensi & Hak Cipta

Proyek ini dirilis di bawah lisensi [MIT License](LICENSE).  
Dipersilakan untuk digunakan, dimodifikasi, dan dikembangkan untuk keperluan akademis maupun komersial.

**Pengembang:** [Fariz Hadi Pamungkas](https://github.com/vanilalatte7) — *AI, IoT & Telecommunication Engineer*
