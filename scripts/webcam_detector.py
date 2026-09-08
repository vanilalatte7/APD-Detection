"""
K3 PPE Real-time Webcam & CCTV Test Script
Dual-stage detection: YOLOv8 Detection + CNN Classification (Helmet & Shoes)
"""

import argparse
import os
import cv2
from ultralytics import YOLO

# Resolve project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_DET = os.path.join(PROJECT_ROOT, "models", "detection", "yolov8_apd_best.pt")
DEFAULT_HELM = os.path.join(PROJECT_ROOT, "models", "classification", "klasifikasi_helm.pt")
DEFAULT_SHOES = os.path.join(PROJECT_ROOT, "models", "classification", "klasifikasi_sepatu.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="K3 PPE Real-time Detection & Classification")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, RTSP URL, or video path)")
    parser.add_argument("--model-det", type=str, default=DEFAULT_DET, help="Path to YOLO detection model")
    parser.add_argument("--model-helm", type=str, default=DEFAULT_HELM, help="Path to CNN helmet classifier")
    parser.add_argument("--model-shoes", type=str, default=DEFAULT_SHOES, help="Path to CNN shoes classifier")
    parser.add_argument("--conf-det", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--conf-cls", type=float, default=0.6, help="Classification confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Loading K3 Detection & Classification Models...")
    try:
        model_det = YOLO(args.model_det)
        model_cls_helm = YOLO(args.model_helm) if os.path.exists(args.model_helm) else None
        model_cls_shoes = YOLO(args.model_shoes) if os.path.exists(args.model_shoes) else None
        print("[OK] Models loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return

    # Parse video source (int if single digit)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        return

    print("[INFO] Video stream started. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video stream ended or frame unavailable.")
            break

        annotated_frame = frame.copy()
        results = model_det(frame, conf=args.conf_det, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                label = model_det.names[int(box.cls[0])].lower()

                color = (0, 255, 0)
                status_text = label.upper()

                # Helmet verification
                if ("helm" in label or "helmet" in label) and model_cls_helm:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        res = model_cls_helm(crop, verbose=False)
                        pred_name = res[0].names[res[0].probs.top1].lower()
                        conf = res[0].probs.top1conf.item()

                        if "not" in pred_name or "no" in pred_name or "tanpa" in pred_name:
                            color = (0, 0, 255)
                            status_text = f"TANPA HELM ({conf:.1%})"
                        else:
                            color = (0, 255, 0)
                            status_text = f"HELM STANDAR ({conf:.1%})"

                # Safety Shoes verification
                elif ("sepatu" in label or "shoes" in label) and model_cls_shoes:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        res = model_cls_shoes(crop, verbose=False)
                        pred_name = res[0].names[res[0].probs.top1].lower()
                        conf = res[0].probs.top1conf.item()

                        if "not" in pred_name or "no" in pred_name or "tanpa" in pred_name:
                            color = (0, 0, 255)
                            status_text = f"TANPA SEPATU ({conf:.1%})"
                        else:
                            color = (0, 255, 0)
                            status_text = f"SEPATU SAFETY ({conf:.1%})"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    status_text,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        cv2.imshow("K3 PPE Real-time Monitor (Press Q to quit)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
