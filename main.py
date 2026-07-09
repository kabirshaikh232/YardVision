from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np
import cv2
import random
import os
import uuid
import gc
from datetime import datetime

from app.detector import detect_objects, detect_and_annotate
from app.database import init_db, get_db, Detection
from app.schemas import DetectionResponse, StatusResponse
from app.video_processor import process_video
from app.gps_simulator import start_gps_simulation, get_all_live_positions, simulate_gps_update, ZONE_BOUNDARIES
from app.camera import start_camera_stream, stop_camera_stream, get_camera_state, CAMERA_URL

app = FastAPI(title="Yard Vision API", version="1.0.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

ZONES = ["Zone-A", "Zone-B", "Zone-C", "Zone-D"]


@app.on_event("startup")
def startup_event():
    init_db()
    os.makedirs("data", exist_ok=True)
    start_gps_simulation()
    print("Database initialized")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("app/static/index.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse("app/static/index.html")


@app.post("/detect")
async def detect(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload an image and detect vehicles/containers."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    detections, annotated_image, counts = detect_and_annotate(frame)

    saved = []
    for det in detections:
        zone = random.choice(ZONES)
        record = Detection(
            class_name=det["class"],
            confidence=det["confidence"],
            ocr_text=det["ocr_text"],
            zone=zone,
            latitude=round(random.uniform(16.85, 16.90), 6),
            longitude=round(random.uniform(74.55, 74.60), 6),
            timestamp=datetime.utcnow()
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        saved.append({
            "id": record.id,
            "class": record.class_name,
            "confidence": record.confidence,
            "ocr_text": record.ocr_text,
            "zone": record.zone,
            "latitude": record.latitude,
            "longitude": record.longitude,
            "timestamp": str(record.timestamp)
        })

    return {
        "total_detected": len(saved),
        "vehicle_counts": counts,
        "detections": saved,
        "annotated_image": annotated_image
    }


@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a video and detect vehicles frame by frame."""
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    video_path = f"data/{unique_name}"

    try:
        with open(video_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"File upload failed: {str(e)}"})

    try:
        result = process_video(video_path, frame_interval=30)
    except Exception as e:
        try:
            gc.collect()
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

    if "error" in result:
        try:
            gc.collect()
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
        return JSONResponse(status_code=400, content=result)

    saved = []
    for det in result["detections"]:
        zone = random.choice(ZONES)
        record = Detection(
            class_name=det["class"],
            confidence=det["confidence"],
            ocr_text=det["ocr_text"],
            zone=zone,
            latitude=round(random.uniform(16.85, 16.90), 6),
            longitude=round(random.uniform(74.55, 74.60), 6),
            timestamp=datetime.utcnow()
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        saved.append({
            "id": record.id,
            "class": record.class_name,
            "confidence": record.confidence,
            "ocr_text": record.ocr_text,
            "zone": record.zone,
            "latitude": record.latitude,
            "longitude": record.longitude,
            "frame": det.get("frame"),
            "time_sec": det.get("time_sec"),
            "timestamp": str(record.timestamp)
        })

    try:
        gc.collect()
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception:
        pass

    return {
        "video_info": {
            "duration_sec": result["duration_sec"],
            "total_frames": result["total_frames"],
            "processed_frames": result["processed_frames"],
            "fps": result["fps"]
        },
        "total_detected": len(saved),
        "detections": saved
    }


@app.get("/camera/status")
def camera_status():
    """Get current camera stream status."""
    state = get_camera_state()
    return {
        "running": state["running"],
        "error": state["error"],
        "has_frame": state["latest_frame"] is not None,
        "has_detection": state["latest_annotated"] is not None,
        "latest_counts": state["latest_counts"],
        "total_detections": len(state["latest_detections"]),
        "camera_url": CAMERA_URL
    }


@app.post("/camera/start")
def camera_start(db: Session = Depends(get_db)):
    """Start live camera stream with AI detection."""
    state = get_camera_state()
    if state["running"]:
        return {"message": "Camera already running"}
    start_camera_stream()
    return {"message": "Camera stream started", "url": CAMERA_URL}


@app.post("/camera/stop")
def camera_stop():
    """Stop live camera stream."""
    stop_camera_stream()
    return {"message": "Camera stream stopped"}


@app.get("/camera/frame")
def camera_frame():
    """Get latest camera frame with detections."""
    state = get_camera_state()
    if not state["running"]:
        return JSONResponse(status_code=400, content={"error": "Camera not running"})
    return {
        "running": state["running"],
        "frame": state["latest_annotated"] or state["latest_frame"],
        "detections": state["latest_detections"],
        "counts": state["latest_counts"],
        "error": state["error"]
    }


@app.post("/camera/capture")
def camera_capture(db: Session = Depends(get_db)):
    """Capture current frame and save detections to DB."""
    state = get_camera_state()
    if not state["running"] or not state["latest_detections"]:
        return JSONResponse(status_code=400, content={"error": "No detections available"})

    saved = []
    for det in state["latest_detections"]:
        zone = random.choice(ZONES)
        record = Detection(
            class_name=det["class"],
            confidence=det["confidence"],
            ocr_text=det["ocr_text"],
            zone=zone,
            latitude=round(random.uniform(16.85, 16.90), 6),
            longitude=round(random.uniform(74.55, 74.60), 6),
            timestamp=datetime.utcnow()
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        saved.append({
            "id": record.id,
            "class": record.class_name,
            "confidence": record.confidence,
            "ocr_text": record.ocr_text,
            "zone": record.zone,
            "timestamp": str(record.timestamp)
        })

    return {
        "total_captured": len(saved),
        "detections": saved
    }


@app.get("/gps/live")
def get_live_gps():
    """Get live GPS positions of all vehicles."""
    positions = get_all_live_positions()
    return {
        "total_vehicles": len(positions),
        "vehicles": list(positions.values()),
        "zone_boundaries": ZONE_BOUNDARIES
    }


@app.get("/gps/zones")
def get_zone_summary():
    """Get vehicle count per zone from live GPS."""
    positions = get_all_live_positions()
    zone_counts = {}
    for vehicle in positions.values():
        zone = vehicle["zone"]
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
    return {
        "zones": [
            {"zone": zone, "vehicle_count": count}
            for zone, count in zone_counts.items()
        ]
    }


@app.get("/detections")
def get_detections(db: Session = Depends(get_db)):
    """Get all recent detections."""
    results = db.query(Detection).order_by(Detection.timestamp.desc()).limit(50).all()
    return {
        "detections": [
            {
                "id": d.id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "ocr_text": d.ocr_text,
                "zone": d.zone,
                "latitude": d.latitude,
                "longitude": d.longitude,
                "timestamp": str(d.timestamp)
            }
            for d in results
        ]
    }


@app.delete("/detections")
def clear_detections(db: Session = Depends(get_db)):
    """Clear all detections from database."""
    db.query(Detection).delete()
    db.commit()
    return {"message": "All detections cleared"}


@app.delete("/reset")
def reset_database(db: Session = Depends(get_db)):
    """Reset entire database."""
    db.query(Detection).delete()
    db.commit()
    return {"message": "Database reset complete"}


@app.get("/inventory")
def get_inventory(db: Session = Depends(get_db)):
    """Get current inventory summary grouped by class and zone."""
    results = (
        db.query(
            Detection.class_name,
            Detection.zone,
            func.count(Detection.id).label("count"),
            func.max(Detection.timestamp).label("last_seen")
        )
        .group_by(Detection.class_name, Detection.zone)
        .all()
    )
    inventory = [
        {
            "class": r.class_name,
            "zone": r.zone,
            "count": r.count,
            "last_seen": str(r.last_seen)
        }
        for r in results
    ]
    return {"inventory": inventory}


@app.get("/analytics")
def get_analytics(db: Session = Depends(get_db)):
    """Get analytics data for charts."""
    zone_data = (
        db.query(Detection.zone, func.count(Detection.id).label("count"))
        .group_by(Detection.zone)
        .all()
    )
    type_data = (
        db.query(Detection.class_name, func.count(Detection.id).label("count"))
        .group_by(Detection.class_name)
        .all()
    )
    hourly_data = (
        db.query(
            func.strftime('%H:00', Detection.timestamp).label("hour"),
            func.count(Detection.id).label("count")
        )
        .group_by(func.strftime('%H:00', Detection.timestamp))
        .order_by("hour")
        .limit(24)
        .all()
    )
    return {
        "zones": [{"zone": r.zone, "count": r.count} for r in zone_data],
        "types": [{"type": r.class_name, "count": r.count} for r in type_data],
        "hourly": [{"hour": r.hour, "count": r.count} for r in hourly_data]
    }


@app.get("/status", response_model=StatusResponse)
def get_status(db: Session = Depends(get_db)):
    """Get system status."""
    total = db.query(Detection).count()
    zones = [z[0] for z in db.query(Detection.zone).distinct().all() if z[0]]
    return StatusResponse(
        status="running",
        total_detections=total,
        active_zones=zones
    )