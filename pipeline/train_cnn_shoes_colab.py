# =================================================================
# SKRIP PELATIHAN CNN STANDAR INDUSTRI (GOOGLE COLAB) - VERSI FINAL
# Proyek: Klasifikasi Jenis Sepatu (Safety vs Tidak_Safety)
# Arsitektur: EfficientNetB0 (Transfer Learning)
# =================================================================

# --- CELL 1: Persiapan Environment, Unzip & Pembersihan ---
import os
import zipfile
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# 1. Konfigurasi Path
zip_path = 'CNN_Sepatu.v1i.folder.zip'
extract_path = './dataset_sepatu'

# 2. Unzip Dataset
if os.path.exists(zip_path):
    # Bersihkan folder lama jika ada
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"✅ Dataset berhasil di-unzip ke {extract_path}")
else:
    print("❌ ERROR: File CNN_Sepatu.v1i.folder.zip tidak ditemukan. Silakan upload file ke Colab!")

# 3. Pembersihan Folder Sampah (Hidden folders & Unlabeled)
garbage_folders = ['.ipynb_checkpoints', '__MACOSX', '.DS_Store']
for root, dirs, files in os.walk(extract_path):
    for d in dirs:
        if d in garbage_folders:
            shutil.rmtree(os.path.join(root, d))
            print(f"🗑️ Menghapus folder sistem: {d}")

# --- CELL 2: Konfigurasi Parameter & Dataset ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = extract_path

# Tentukan target kelas secara eksplisit sesuai kebutuhan industri
TARGET_CLASSES = ['Safety', 'Tidak_Safety']

# Deteksi struktur folder (Train/Valid)
train_dir = os.path.join(DATA_DIR, 'train')
val_dir = os.path.join(DATA_DIR, 'valid')

# Jika tidak ada sub-folder train/valid (semua di root), gunakan split otomatis
if not os.path.exists(train_dir):
    train_dir = DATA_DIR
    val_dir = None
    print("ℹ️ Menggunakan folder utama dengan split otomatis 20% untuk validasi.")

# --- CELL 3: Data Augmentation & Generators ---
# Note: EfficientNetB0 di tf.keras sudah memiliki layer rescaling internal (0-255)
# Namun menggunakan rescale=1./255 tetap aman dan standar.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 if val_dir is None else 0.0
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Generator Train
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=TARGET_CLASSES, # HANYA ambil kelas yang diinginkan
    subset='training' if val_dir is None else None,
    shuffle=True
)

# Generator Validasi
validation_generator = (val_datagen if val_dir else train_datagen).flow_from_directory(
    val_dir if val_dir else train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=TARGET_CLASSES, # HANYA ambil kelas yang diinginkan
    subset='validation' if val_dir is None else None,
    shuffle=False
)

num_classes = len(TARGET_CLASSES)
print(f"✅ Konfigurasi Selesai. Siap melatih {num_classes} kelas: {TARGET_CLASSES}")

# --- CELL 4: Arsitektur Model (EfficientNetB0) ---
print("🏗️ Membangun model standar industri...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Tahap 1: Pembekuan base model
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x) # Dropout lebih tinggi untuk dataset kecil agar tidak overfit
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- CELL 5: Training Tahap 1 (Head Only) ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_shoe_model_v2.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("🚀 Memulai Pelatihan Tahap 1...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,
    callbacks=callbacks
)

# --- CELL 6: Fine-Tuning (Membuka Layer Atas) ---
print("🔧 Memulai Fine-Tuning untuk akurasi maksimal...")
base_model.trainable = True
# Bekukan semua kecuali 30 layer terakhir
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Learning rate sangat kecil
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,
    callbacks=callbacks
)

# --- CELL 7: Evaluasi & Plot ---
def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    
    plt.figure(figsize=(8, 5))
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Akurasi Model Final')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(history, history_fine)
print("🎉 SELESAI! Model terbaik disimpan sebagai 'best_shoe_model_v2.h5'")
