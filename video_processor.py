import cv2
import numpy as np
from app.detector import detect_objects

def process_video(video_path: str, frame_interval: int = 30):
    """
    Process a video file frame by frame.
    frame_interval: process every Nth frame (30 = every 1 sec at 30fps)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Could not open video file"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = round(total_frames / fps, 2) if fps > 0 else 0

    all_detections = []
    frame_number = 0
    processed_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                detections = detect_objects(frame)
                for det in detections:
                    det["frame"] = frame_number
                    det["time_sec"] = round(frame_number / fps, 2) if fps > 0 else 0
                    all_detections.append(det)
                processed_frames += 1

            frame_number += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return {
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "fps": fps,
        "duration_sec": duration,
        "detections": all_detections
    }