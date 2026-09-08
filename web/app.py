"""
K3 PPE Violation Detection & Real-time Monitoring Server
Two-stage AI Pipeline: YOLOv8 Object Detection + CNN Helmet/Shoe Classifier
Integrated with Flask, MySQL, and automated violation evidence logging.
"""

from flask import Flask, render_template, Response, session, redirect, url_for, request
import cv2
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
import secrets
import os
import time
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))

# --- PATH RESOLUTION ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- VIDEO SOURCE CONFIGURATION ---
raw_video_source = os.getenv("VIDEO_SOURCE", "0")
if raw_video_source.isdigit():
    VIDEO_SOURCE = int(raw_video_source)
else:
    if not os.path.isabs(raw_video_source):
        possible_path = os.path.join(BASE_DIR, raw_video_source)
        VIDEO_SOURCE = possible_path if os.path.exists(possible_path) else raw_video_source
    else:
        VIDEO_SOURCE = raw_video_source

# --- DATABASE CONFIGURATION ---
DB_URI = os.getenv("DATABASE_URL", "mysql+mysqlconnector://root:@localhost/k3monitoring")
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- DATABASE MODELS ---
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100))
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(100))

class Pelanggaran(db.Model):
    __tablename__ = 'history_pelanggaran'
    id = db.Column(db.Integer, primary_key=True)
    jenis_pelanggaran = db.Column(db.String(100))
    keterangan = db.Column(db.String(255))
    bukti = db.Column(db.String(100))
    waktu = db.Column(db.String(50))
    akurasi = db.Column(db.String(20))

with app.app_context():
    try:
        db.create_all()
    except Exception as db_err:
        print(f"[WARN] Database initialization warning (ensure MySQL is running): {db_err}")

# --- MODEL PATHS RESOLUTION ---
DEFAULT_DET_PATH = os.path.join(PROJECT_ROOT, "models", "detection", "yolov8_apd_best.pt")
DEFAULT_CLS_SEPATU = os.path.join(PROJECT_ROOT, "models", "classification", "klasifikasi_sepatu.pt")
DEFAULT_CLS_HELM = os.path.join(PROJECT_ROOT, "models", "classification", "klasifikasi_helm.pt")

MODEL_PATH_DET = os.getenv("MODEL_PATH_DET", DEFAULT_DET_PATH)
MODEL_PATH_CLS_SEPATU = os.getenv("MODEL_PATH_CLS_SEPATU", DEFAULT_CLS_SEPATU)
MODEL_PATH_CLS_HELM = os.getenv("MODEL_PATH_CLS_HELM", DEFAULT_CLS_HELM)

model_det = None
model_cls_sepatu = None
model_cls_helm = None

try:
    if os.path.exists(MODEL_PATH_DET):
        model_det = YOLO(MODEL_PATH_DET)
        print(f"[OK] YOLO Detection Model loaded: {MODEL_PATH_DET}")
    else:
        print(f"[WARN] YOLO model not found at {MODEL_PATH_DET}")

    if os.path.exists(MODEL_PATH_CLS_SEPATU):
        model_cls_sepatu = YOLO(MODEL_PATH_CLS_SEPATU)
        print(f"[OK] Shoes Classifier loaded: {MODEL_PATH_CLS_SEPATU}")

    if os.path.exists(MODEL_PATH_CLS_HELM):
        model_cls_helm = YOLO(MODEL_PATH_CLS_HELM)
        print(f"[OK] Helmet Classifier loaded: {MODEL_PATH_CLS_HELM}")
except Exception as e:
    print(f"[ERROR] Error loading models: {e}")

last_capture_time = 0

# --- HTTP ROUTES ---
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html', page='login')

@app.route('/login', methods=['POST'])
def login():
    u, p = request.form.get('username'), request.form.get('password')
    try:
        user = User.query.filter_by(username=u, password=p).first()
        if user:
            session['username'], session['nama'] = user.username, user.nama
            return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Login error: {e}")
    return "Gagal Login! <a href='/'>Kembali</a>"

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nama, user, pw = request.form.get('nama'), request.form.get('username'), request.form.get('password')
        try:
            if User.query.filter_by(username=user).first():
                return "Username sudah ada! <a href='/register'>Kembali</a>"
            new_user = User(nama=nama, username=user, password=pw)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('index'))
        except Exception as e:
            return f"Error pendaftaran: {e} <a href='/register'>Kembali</a>"
    return render_template('index.html', page='register')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('index.html', page='monitor', nama=session.get('nama'))

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('index'))
    try:
        daftar_pelanggaran = Pelanggaran.query.order_by(Pelanggaran.id.desc()).all()
    except Exception as e:
        print(f"History fetch error: {e}")
        daftar_pelanggaran = []
    return render_template('index.html', page='history', data=daftar_pelanggaran)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# --- CORE LOGIC: DETEKSI & AUTO CAPTURE ---
def generate_frames():
    global last_capture_time

    print(f"[INFO] Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {VIDEO_SOURCE}")
        return
    else:
        print(f"[OK] Video source opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    while True:
        success, frame = cap.read()

        if not success:
            if isinstance(VIDEO_SOURCE, str) and os.path.exists(VIDEO_SOURCE):
                # Loop video file automatically
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        if model_det:
            results = model_det(frame, conf=0.5, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label_idx = int(box.cls[0])
                    label = model_det.names[label_idx].lower()

                    color, txt = (0, 255, 0), label.upper()
                    is_violation = False

                    crop = frame[max(0, y1):y2, max(0, x1):x2]

                    if crop.size > 0:
                        try:
                            # Verifikasi Helm
                            if ("helm" in label or "helmet" in label) and model_cls_helm:
                                res_h = model_cls_helm(crop, verbose=False)
                                c_lab = res_h[0].names[res_h[0].probs.top1].lower()
                                if "not" in c_lab or "no" in c_lab or "tanpa" in c_lab:
                                    is_violation, color = True, (0, 0, 255)
                                    txt = "TANPA HELM"

                            # Verifikasi Sepatu
                            elif ("sepatu" in label or "shoes" in label) and model_cls_sepatu:
                                res_s = model_cls_sepatu(crop, verbose=False)
                                c_lab = res_s[0].names[res_s[0].probs.top1].lower()
                                if "not" in c_lab or "no" in c_lab or "tanpa" in c_lab:
                                    is_violation, color = True, (0, 0, 255)
                                    txt = "TANPA SEPATU"
                        except Exception as e:
                            print(f"[WARN] Classification Error: {e}")

                    # Draw Bounding Box & Label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, txt, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Auto Capture Violation Evidence
                    now = time.time()
                    if is_violation and (now - last_capture_time > 7):
                        last_capture_time = now
                        waktu_foto = datetime.now()
                        file_name = f"cap_{waktu_foto.strftime('%Y%m%d_%H%M%S')}.jpg"
                        full_path = os.path.join(UPLOAD_FOLDER, file_name)

                        if cv2.imwrite(full_path, frame):
                            print(f"[ALERT] Violation captured: {file_name}")
                            try:
                                with app.app_context():
                                    new_log = Pelanggaran(
                                        jenis_pelanggaran=label.upper(),
                                        keterangan=f"Terdeteksi {txt}",
                                        bukti=file_name,
                                        waktu=waktu_foto.strftime("%Y-%m-%d %H:%M:%S"),
                                        akurasi=f"{box.conf[0].item():.1%}"
                                    )
                                    db.session.add(new_log)
                                    db.session.commit()
                            except Exception as save_err:
                                print(f"[WARN] Database log failed: {save_err}")

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"Starting K3 APD Monitoring Server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
