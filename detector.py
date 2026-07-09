import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
import base64

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Load EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Target classes
TARGET_CLASSES = ["truck", "car", "bus", "train"]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.50

# Colors per class (BGR)
CLASS_COLORS = {
    "truck": (0, 212, 255),
    "car": (0, 255, 136),
    "bus": (255, 136, 255),
    "train": (255, 200, 0),
}


def detect_objects(frame: np.ndarray):
    """Run YOLOv8 detection on a single frame."""
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name not in TARGET_CLASSES:
            continue
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]
        ocr_text = extract_text(cropped)

        detections.append({
            "class": class_name,
            "confidence": round(confidence, 2),
            "bbox": [x1, y1, x2, y2],
            "ocr_text": ocr_text
        })

    return detections


def detect_and_annotate(frame: np.ndarray):
    """
    Run detection and return:
    - detections list
    - annotated image as base64 string
    - vehicle counts per class
    """
    detections = detect_objects(frame)
    annotated = frame.copy()
    counts = {}

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class"]
        confidence = det["confidence"]
        ocr_text = det["ocr_text"]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        # Count vehicles
        counts[class_name] = counts.get(class_name, 0) + 1

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{class_name} {int(confidence*100)}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 6, y1), color, -1)

        # Draw label text
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Draw OCR text below box if available
        if ocr_text:
            ocr_label = f"ID: {ocr_text[:20]}"
            cv2.putText(annotated, ocr_label, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw summary counter on top left
    y_offset = 30
    for cls, count in counts.items():
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.putText(annotated, f"{cls}: {count}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        y_offset += 28

    # Convert to base64
    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return detections, img_base64, counts


def extract_text(image: np.ndarray) -> str:
    """Run EasyOCR on a cropped image region."""
    if image.size == 0:
        return ""
    results = reader.readtext(image)
    texts = [text for (_, text, confidence) in results if confidence > 0.3]
    return " ".join(texts).strip()