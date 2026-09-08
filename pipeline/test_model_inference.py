import cv2
from ultralytics import YOLO

# --- 1. INISIALISASI MODEL ---
model_path = r"D:\Fari Hadi P\yusha\Pengolahan Dataset\model yolo\best (11).pt"
video_path = r"D:\Fari Hadi P\yusha\Pengolahan Dataset\lv_0_20260427043807.mp4" 
output_path = "hasil_deteksi_custom.mp4"
model = YOLO(model_path)

# CEK CLASS ID! (Lihat output terminal Anda)
print("\n" + "="*50)
print("DAFTAR CLASS MODEL ANDA:", model.names)
print("="*50 + "\n")

# --- 2. UBAH ANGKA INI SESUAI OUTPUT TERMINAL DI ATAS ---
ID_ORANG = 1   # Ganti jika person bukan 0
ID_HELM = 0    # Ganti jika helmet bukan 1
ID_SEPATU = 2  # Ganti jika shoes bukan 2

# Fungsi bantuan untuk mengecek apakah dua kotak saling bersinggungan
def is_overlapping(box1, box2):
    # box format: [x1, y1, x2, y2]
    # Jika salah satu kotak berada sepenuhnya di kiri, kanan, atas, atau bawah kotak lainnya = tidak bersinggungan
    if box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3]:
        return False
    return True

print("🎬 Memulai proses video...")

cap = cv2.VideoCapture(video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Turunkan confidence sedikit jika model kurang sensitif (misal: conf=0.20)
    results = model.predict(source=frame, conf=0.25, verbose=False)
    boxes = results[0].boxes
    
    orang_list = []
    helm_list = []
    sepatu_list = []

    # Kelompokkan hasil deteksi
    for box in boxes:
        cls_id = int(box.cls[0])
        coords = box.xyxy[0].tolist() 
        
        if cls_id == ID_ORANG:
            orang_list.append(coords)
        elif cls_id == ID_HELM:
            helm_list.append(coords)
        elif cls_id == ID_SEPATU:
            sepatu_list.append(coords)

    # Cek kelengkapan untuk setiap orang
    for o_coords in orang_list:
        pakai_helm = False
        pakai_sepatu = False
        
        # Cek persinggungan dengan helm
        for h_coords in helm_list:
            if is_overlapping(o_coords, h_coords):
                pakai_helm = True
                break
                
        # Cek persinggungan dengan sepatu
        for s_coords in sepatu_list:
            if is_overlapping(o_coords, s_coords):
                pakai_sepatu = True
                break

        # Tentukan Status
        if pakai_helm and pakai_sepatu:
            label = "Lengkap"
            color = (0, 255, 0) # Hijau
        else:
            label = "Tidak Lengkap"
            color = (0, 0, 255) # Merah

        # Gambar Bounding Box
        ox1, oy1, ox2, oy2 = [int(x) for x in o_coords]
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 3)
        cv2.rectangle(frame, (ox1, oy1 - 30), (ox1 + 180, oy1), color, -1)
        cv2.putText(frame, label, (ox1 + 5, oy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Hanya untuk debugging: print jumlah orang yang terdeteksi tiap 30 frame
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}: Terdeteksi {len(orang_list)} orang.")

    cv2.imshow("Uji Coba Deteksi APD", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n🎉 SELESAI!")