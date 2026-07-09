import cv2
import base64
import threading
import numpy as np
from app.detector import detect_and_annotate

CAMERA_URL = "http://10.194.223.22:8080/video"

# Shared state
camera_state = {
    "running": False,
    "latest_frame": None,
    "latest_annotated": None,
    "latest_detections": [],
    "latest_counts": {},
    "error": None
}
_lock = threading.Lock()


def frame_to_base64(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')


def start_camera_stream(camera_url: str = CAMERA_URL):
    """Start background thread to read camera and run detection."""
    def run():
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            with _lock:
                camera_state["error"] = "Cannot open camera stream"
                camera_state["running"] = False
            return

        with _lock:
            camera_state["running"] = True
            camera_state["error"] = None

        frame_count = 0
        while camera_state["running"]:
            ret, frame = cap.read()
            if not ret:
                with _lock:
                    camera_state["error"] = "Stream disconnected"
                break

            # Store raw frame as base64 always
            with _lock:
                camera_state["latest_frame"] = frame_to_base64(frame)

            # Run detection every 10th frame to save CPU
            if frame_count % 10 == 0:
                try:
                    detections, annotated_b64, counts = detect_and_annotate(frame)
                    with _lock:
                        camera_state["latest_annotated"] = annotated_b64
                        camera_state["latest_detections"] = detections
                        camera_state["latest_counts"] = counts
                except Exception as e:
                    pass

            frame_count += 1

        cap.release()
        with _lock:
            camera_state["running"] = False

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


def stop_camera_stream():
    with _lock:
        camera_state["running"] = False


def get_camera_state():
    with _lock:
        return dict(camera_state)