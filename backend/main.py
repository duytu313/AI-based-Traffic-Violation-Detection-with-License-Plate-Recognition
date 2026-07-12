"""
Traffic AI Backend - FastAPI Server
Replaces Streamlit with Next.js frontend + FastAPI backend
"""
import os
import sys
import cv2
import time
import json
from datetime import datetime
import numpy as np
import tempfile
import threading
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Add parent directory to path so we can import database.py and src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from database import (
    init_db, insert_vehicle_entry, update_exit_vehicle, insert_violation, get_connection,
    get_global_stats,
    # Parking
    get_parking_stats, get_parking_entries, get_parking_slots, insert_parking_entry, update_parking_exit,
    # Logistics
    get_logistics_stats, get_logistics_entries, get_unknown_alerts, get_truck_visits,
    insert_logistics_entry, update_logistics_exit, insert_unknown_alert,
    # SmartCity
    get_smartcity_stats, get_traffic_flow, get_city_violations, get_flow_by_hour,
    insert_traffic_flow, insert_city_violation, insert_red_light_violation,
    # Test
    test_get_vehicles, test_get_violations, test_get_stats,
    # Speed violations
    db_insert_speed_violation, db_get_speed_violations, db_get_speed_violation_stats,
)
from backend.src.core.tracker import ByteTrackVehicleTracker, VideoCaptureThread
from backend.src.utils.image_utils import (
    crop_vehicle_context, build_tracking_dets, find_vehicle_context_by_bbox,
    build_hierarchical_violation_map, merge_violation_data
)
from backend.src.utils.notifications import send_telegram_notification, send_violation_telegram
from backend.src.utils.device_utils import get_device
# Report generator removed - functions will be added back if needed

# Import models directly (without Streamlit decorator)
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer as FastPlateOCR

# Initialize database
init_db()

# Load models (direct loading for backend - no Streamlit)
print("Loading AI models...")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files_model")
yolo_plate = YOLO(os.path.join(MODEL_DIR, "license_plate_detector.pt"))
yolo_vehicle = YOLO(os.path.join(MODEL_DIR, "yolov8n.pt"))
try:
    color_model = YOLO(os.path.join(MODEL_DIR, "vehicle_color_n_cls.pt"))
except Exception as e:
    print(f"Warning: Cannot load color model: {e}")
    color_model = None
try:
    helmet_model = YOLO(os.path.join(MODEL_DIR, "helmet.pt"))
except Exception as e:
    print(f"Warning: Cannot load helmet model: {e}")
    helmet_model = None
try:
    traffic_light_model = YOLO(os.path.join(MODEL_DIR, "traffic_light.pt"))
except Exception as e:
    print(f"Warning: Cannot load traffic light model: {e}")
    traffic_light_model = None
ocr_engine = FastPlateOCR(hub_ocr_model='cct-s-v2-global-model', device='auto')

# Move models to GPU if available
_main_device = get_device()
if _main_device != 'cpu':
    try:
        yolo_plate.to(_main_device)
        yolo_vehicle.to(_main_device)
        if color_model is not None:
            color_model.to(_main_device)
        if helmet_model is not None:
            helmet_model.to(_main_device)
        if traffic_light_model is not None:
            traffic_light_model.to(_main_device)
        print(f"[Device] Models moved to {_main_device}")
    except Exception as e:
        print(f"Warning: Could not move models to {_main_device}: {e}")

from backend.src.core.engine import LicensePlateRecognizer as LPR
from backend.src.core.zones import ZoneConfig
from backend.src.core.roi_detector import ROIDetector, ROIConfig
from backend.src.core.birds_eye_detector import BirdsEyeRedLightDetector, BEVConfig
from backend.src.core.ocr_consolidator import OCRConsolidator
# Monkey-patch LicensePlateRecognizer for non-Streamlit env
class LicensePlateRecognizer(LPR):
    def detect_vehicles(self, image, debug=False):
        from ultralytics import YOLO
        _device = get_device()
        results = self.yolo_vehicle.predict(
            image, device=_device, classes=list(self.vehicle_classes.keys()),
            conf=self.vehicle_conf
        )[0]
        vehicles = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                vehicle_type = self.vehicle_classes.get(cls_id, "vehicle")
                from backend.src.utils.image_utils import crop_vehicle_context
                vehicle_crop, crop_bbox = crop_vehicle_context(
                    image, (x1, y1, x2, y2), vehicle_type
                )
                vehicles.append(((x1, y1, x2, y2), vehicle_type, "unknown", conf, vehicle_crop, crop_bbox))
                color = self.classify_color(vehicle_crop)
                if len(vehicles[-1]) == 6:
                    vehicles[-1] = (vehicles[-1][0], vehicles[-1][1], color,
                                    vehicles[-1][3], vehicles[-1][4], vehicles[-1][5])
                else:
                    vehicles[-1] = (vehicles[-1][0], vehicles[-1][1], color,
                                    vehicles[-1][3], vehicles[-1][4])
        return vehicles

recognizer = LicensePlateRecognizer(
    yolo_plate, yolo_vehicle, color_model, helmet_model,
    traffic_light_model, ocr_engine, vehicle_conf=0.25, plate_conf=0.5  # Increased to 0.5 to reduce false plate detections
)
# Lower thresholds for better detection
recognizer.violation_conf_using_mobile = 0.15  # Lower threshold for mobile detection
recognizer.violation_conf_without_helmet = 0.20
recognizer.violation_conf_more_than_two = 0.40

# Initialize ROI detector
roi_detector = ROIDetector()

# Initialize OCR Consolidator for frame-level plate voting
# min_vote_frames=5: wait for at least 5 OCR reads before finalizing
# vote_ratio=0.6: require 60% majority for consensus
# track_timeout=10.0: remove tracks unseen for 10 seconds
ocr_consolidator = OCRConsolidator(
    min_vote_frames=5,
    vote_ratio=0.6,
    track_timeout=10.0,
    cleanup_interval=30.0
)

print("Models loaded successfully!")

app = FastAPI(title="Traffic AI API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve evidence images
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(os.path.join(DATA_DIR, "vehicles"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "plates"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "evidence"), exist_ok=True)
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")


# ======================== MODELS ========================

class ProcessImageResponse(BaseModel):
    vehicles: list
    violations: list
    red_light_violations: list
    stats: dict
    image_base64: Optional[str] = None

class StatsResponse(BaseModel):
    total_vehicles: int
    fraud_alerts: int
    total_violations: int

class ROIDetectRequest(BaseModel):
    vehicles: List[Dict[str, Any]]
    red_light_active: bool = False



# ======================== API ENDPOINTS ========================

@app.get("/api/health")
def health_check():
    return {"status": "ok", "timestamp": time.time()}


# ======================== ZONE-BASED RED LIGHT DETECTION API ========================

@app.post("/api/zone-config")
def set_zone_config(
    waiting_end: Optional[int] = Form(None),
    stop_start: Optional[int] = Form(None),
    stop_end: Optional[int] = Form(None),
    intersection_start: Optional[int] = Form(None),
    intersection_end: Optional[int] = Form(None),
):
    """
    Update zone configuration for zone-based red light detection.
    
    Parameters:
      - waiting_end: Y pixel of waiting zone end (default: 200)
      - stop_start: Y pixel of stop zone start (default: equal to waiting_end)
      - stop_end: Y pixel of stop zone end (default: 300)
      - intersection_start: Y pixel of intersection start (default: equal to stop_end)
      - intersection_end: Y pixel of intersection end (default: 500)
    """
    kwargs = {}
    if waiting_end is not None:
        kwargs['waiting_end'] = waiting_end
    if stop_start is not None:
        kwargs['stop_start'] = stop_start
    if stop_end is not None:
        kwargs['stop_end'] = stop_end
    if intersection_start is not None:
        kwargs['intersection_start'] = intersection_start
    if intersection_end is not None:
        kwargs['intersection_end'] = intersection_end
    
    recognizer.set_zone_config(**kwargs)
    stats = recognizer.get_zone_stats()
    return {"status": "ok", "zone_config": stats["zone_config"]}

@app.get("/api/zone-config")
def get_zone_config():
    """Get current zone configuration."""
    stats = recognizer.get_zone_stats()
    return {"zone_config": stats["zone_config"]}

@app.get("/api/zone-stats")
def get_zone_stats():
    """Get zone detector statistics."""
    return recognizer.get_zone_stats()

@app.post("/api/zone-reset")
def reset_zone_detector():
    """Reset zone detector to initial state."""
    recognizer.reset_zone_detector()
    return {"status": "ok", "message": "Zone detector has been reset"}


# ======================== ROI (REGION OF INTEREST) API ========================

@app.post("/api/roi/set")
async def set_roi(request: Request):
    """
    Update ROI (Region of Interest) zone for violation detection.
    
    Request Body JSON:
        points: List of points [{"x": 100, "y": 200}, ...]
        name: ROI zone name (optional, default: "violation_zone")
    """
    data = await request.json()
    points = data.get("points", [])
    name = data.get("name", "violation_zone")
    # Convert to int for OpenCV compatibility
    points_int = [{"x": int(p["x"]), "y": int(p["y"])} for p in points]
    roi_detector.set_roi(points_int, name)
    config = roi_detector.get_config()
    return {"status": "ok", "roi_config": config}

@app.get("/api/roi/get")
def get_roi():
    """Get current ROI configuration."""
    config = roi_detector.get_config()
    return {"roi_config": config}

@app.post("/api/roi/clear")
def clear_roi():
    """Clear ROI zone."""
    roi_detector.clear_roi()
    return {"status": "ok", "message": "ROI has been cleared"}

@app.post("/api/roi/detect")
def detect_roi_violations(request: ROIDetectRequest):
    """
    Detect violations in ROI.
    
    Args:
        request.body.vehicles: List of vehicles with bbox, track_id, class_name, conf
        request.body.red_light_active: True if red light is active
    """
    violations = roi_detector.process_vehicles(request.vehicles, request.red_light_active)
    return {"violations": violations}


# ======================== BIRD'S EYE VIEW (BEV) RED LIGHT DETECTION API ========================

# Global BEV detector instance
bev_detector = BirdsEyeRedLightDetector()

@app.post("/api/bev/config")
async def set_bev_config(request: Request):
    """
    Update BEV detector configuration.
    
    Request Body JSON:
        src_points: List of 4 source points [[x,y], ...] or [{x:100, y:200}, ...] (trapezoid in image space)
        dst_points: List of 4 destination points [[x,y], ...] or [{x:100, y:200}, ...] (rectangle in BEV space)
        stop_line_3d_y: Stop line position in BEV space (Y coordinate)
        red_light_buffer_frames: Number of buffer frames for red light
    """
    data = await request.json()
    
    if "src_points" in data:
        # Handle both [{x, y}, ...] and [[x, y], ...] formats
        src_points = data["src_points"]
        if src_points and isinstance(src_points[0], dict):
            pts = np.array([[p["x"], p["y"]] for p in src_points], dtype=np.float32)
        else:
            pts = np.array(src_points, dtype=np.float32)
        bev_detector.set_src_points(pts)
    
    if "dst_points" in data:
        # Handle both [{x, y}, ...] and [[x, y], ...] formats
        dst_points = data["dst_points"]
        if dst_points and isinstance(dst_points[0], dict):
            pts = np.array([[p["x"], p["y"]] for p in dst_points], dtype=np.float32)
        else:
            pts = np.array(dst_points, dtype=np.float32)
        bev_detector.set_dst_points(pts)
    
    if "stop_line_3d_y" in data:
        bev_detector.set_stop_line_3d_y(int(data["stop_line_3d_y"]))
    
    if "red_light_buffer_frames" in data:
        bev_detector.set_red_light_buffer_frames(int(data["red_light_buffer_frames"]))
    
    # Invalidate cached frames so BEV zone redraws immediately on all streams
    for stream in [camera_stream] + list(parking_streams.values()) + list(logistics_streams.values()) + list(smartcity_streams.values()):
        with stream.lock:
            stream._last_frame = None
    
    return {"status": "ok", "config": bev_detector.get_stats()}

@app.get("/api/bev/config")
def get_bev_config():
    """Get current BEV detector configuration."""
    return {"config": bev_detector.get_stats()}

@app.post("/api/bev/reset")
def reset_bev_detector():
    """Reset BEV detector to initial state."""
    bev_detector.reset()
    return {"status": "ok", "message": "BEV detector has been reset"}

@app.get("/api/bev/stats")
def get_bev_stats():
    """Get BEV detector statistics."""
    return bev_detector.get_stats()


@app.post("/api/process-image")
async def process_image(
    file: UploadFile = File(...),
    enable_violation_detection: bool = Form(True),
    enable_red_light_detection: bool = Form(False),
    enable_bev_detection: bool = Form(True),
    violation_conf_limit: float = Form(0.15),
    conf_more_than_two: float = Form(0.50),
    conf_no_helmet: float = Form(0.15),
    conf_using_mobile: float = Form(0.15),
    traffic_light_conf: float = Form(0.25),
    debug: bool = Form(False),
    show_zones: bool = Form(False),
    show_bev: bool = Form(True),
    camera_direction: str = Form("down"),
):
    """Process a single image and return detection results."""
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Set config
    recognizer.set_violation_conf_limit(violation_conf_limit)
    recognizer.violation_conf_more_than_two = conf_more_than_two
    recognizer.violation_conf_without_helmet = conf_no_helmet
    recognizer.violation_conf_using_mobile = conf_using_mobile
    recognizer.set_traffic_light_conf(traffic_light_conf)
    
    # Set camera direction for zone-based detection
    recognizer.zone_detector.config.direction = camera_direction

    # Detect vehicles and plates
    vehicles = recognizer.detect_vehicles(image, debug=debug)
    plates = recognizer.detect_plates(image)
    matched = recognizer.match_plates_to_vehicles(vehicles, plates)

    # Deduplicate plates
    unique_matched = []
    seen_plates = set()
    for m in matched:
        if m[4] not in seen_plates:
            unique_matched.append(m)
            seen_plates.add(m[4])
    matched = unique_matched

    # Violation detection
    vehicle_violation_map = {}
    if enable_violation_detection:
        vehicle_violation_map = build_hierarchical_violation_map(recognizer, vehicles, image=image, debug=debug)

    all_violations_by_vehicle = {
        tuple(bbox): {"vtype": vtype, "color": color, "plate_text": plate_text, "violations": list(violations)}
        for bbox, vtype, color, plate_text, violations in merge_violation_data(vehicle_violation_map, vehicles, matched)
    }

    # Always detect traffic lights for drawing frames
    traffic_lights, traffic_road_users = recognizer.detect_traffic_scene(image)
    
    # Red light detection - ROI based (no stop line)
    red_light_violations = []
    if enable_red_light_detection or enable_bev_detection:
        red_light_active = recognizer.red_light_is_active(traffic_lights)
        
        # BEV (Bird's Eye View) red light detection - also uses detect_traffic_scene
        if enable_bev_detection:
            # Update BEV detector with traffic light info
            bev_detector.update_red_light(traffic_lights)
            # Create tracked-like data from vehicles (simulate tracking for single image)
            for idx, v in enumerate(vehicles):
                v_bbox = v[0]
                vx1, vy1, vx2, vy2 = v_bbox
                track_id = hash(tuple(v_bbox)) % 10000
                class_name = v[1]
                bev_result = bev_detector.process_vehicle(track_id, class_name, (vx1, vy1, vx2, vy2))
                if bev_result["first_time_violation"]:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"[{current_time}] 🔴 BEV (Image): Vehicle {bev_result['unique_key'].upper()} crossed 3D boundary at red light!")
                    red_light_violations.append({
                        "bbox": [vx1, vy1, vx2, vy2],
                        "class_name": class_name,
                        "conf": 1.0,
                        "details": f"Red light running (BEV 3D) - {class_name}",
                    })
        
        # Use ROI polygon for violation detection on single image
        if enable_red_light_detection:
            roi_config = roi_detector.get_config()
            if roi_config and red_light_active:
                vehicle_list = []
                for v in vehicles:
                    v_bbox = v[0]
                    vx1, vy1, vx2, vy2 = v_bbox
                    vehicle_list.append({
                        "bbox": [int(vx1), int(vy1), int(vx2), int(vy2)],
                        "track_id": hash(tuple(v_bbox)) % 10000,
                        "class_name": v[1],
                        "conf": float(v[3]),
                    })
                # min_history=1 for single image (no temporal tracking needed)
                roi_results = roi_detector.process_vehicles(vehicle_list, red_light_active=True, min_history=1)
                red_light_violations.extend(roi_results)
            
            # Fallback to legacy zone-based detection if no ROI configured
            if not roi_config and red_light_active:
                zone_results = recognizer.detect_red_light_violations_zone_based(
                    traffic_lights, traffic_road_users, frame=image
                )
                red_light_violations.extend(zone_results)
        
        for viol in red_light_violations:
            vx1, vy1, vx2, vy2 = viol['bbox']
            best_vehicle = None
            best_dist = float('inf')
            for v in vehicles:
                v_bbox = v[0]
                vvx1, vvy1, vvx2, vvy2 = v_bbox
                veh_bottom_center = ((vvx1 + vvx2) / 2.0, vvy2)
                viol_bottom_center = ((vx1 + vx2) / 2.0, vy2)
                dist = np.sqrt(
                    (viol_bottom_center[0] - veh_bottom_center[0]) ** 2 +
                    (viol_bottom_center[1] - veh_bottom_center[1]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_vehicle = v
            if best_vehicle is not None:
                v_bbox = best_vehicle[0]
                plate_text = ""
                for (mb, mt, mc, mp_img, mp_txt) in matched:
                    if mb == v_bbox:
                        plate_text = mp_txt
                        break
                if v_bbox in all_violations_by_vehicle:
                    all_violations_by_vehicle[v_bbox]['violations'].append((
                        'RED_LIGHT_VIOLATION', viol['details'], (vx1, vy1, vx2, vy2), viol['conf']
                    ))
                else:
                    all_violations_by_vehicle[v_bbox] = {
                        'vtype': best_vehicle[1], 'color': best_vehicle[2],
                        'plate_text': plate_text,
                        'violations': [('RED_LIGHT_VIOLATION', viol['details'], (vx1, vy1, vx2, vy2), viol['conf'])]
                    }

    # Build result data
    all_violation_data = []
    if all_violations_by_vehicle:
        for bbox, data in all_violations_by_vehicle.items():
            all_violation_data.append({
                "bbox": list(bbox),
                "vtype": data['vtype'],
                "color": data['color'],
                "plate_text": data['plate_text'],
                "violations": [
                    {"type": v[0], "details": v[1], "bbox": list(v[2]), "conf": v[3]}
                    for v in data['violations']
                ]
            })

    # Draw results on image
    img_draw = image.copy()
    
    # Draw zones if enabled
    if show_zones:
        img_draw = recognizer.draw_traffic_zones(img_draw)
    
    # Draw ROI if enabled
    if roi_detector.get_config() is not None:
        img_draw = roi_detector.draw_roi(img_draw)
    
    # Draw traffic scene (lights + violations) - no stop line
    img_draw = recognizer.draw_traffic_scene(
        img_draw, 
        traffic_lights,
        traffic_road_users,
        red_light_violations,
        stop_line_y=None,  # No stop line - using ROI polygon instead
        show_zones=False  # zones already drawn above
    )
    
    for v in vehicles:
        bbox, vtype, color = v[0], v[1], v[2]
        x1, y1, x2, y2 = bbox
        has_violation = False
        plate_text = ""
        for item in all_violation_data:
            if item['bbox'][0] == x1 and item['bbox'][1] == y1:
                has_violation = True
                plate_text = item['plate_text']
                break
        box_color = (0, 0, 255) if has_violation else (0, 255, 0)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), box_color, 1)
        label = f"{color} {vtype}"
        cv2.putText(img_draw, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
        if plate_text:
            cv2.putText(img_draw, plate_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw license plate bounding boxes
    for plate_img, (px1, py1, px2, py2) in plates:
        cv2.rectangle(img_draw, (px1, py1), (px2, py2), (255, 255, 0), 2)
        cv2.putText(img_draw, "PLATE", (px1, py1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    for item in all_violation_data:
        for viol in item['violations']:
            vx1, vy1, vx2, vy2 = viol['bbox']
            color_v = recognizer.get_violation_color(viol['type'])
            cv2.rectangle(img_draw, (vx1, vy1), (vx2, vy2), color_v, 2)
            label_v = f"{viol['type']} {viol['conf']*100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label_v, font, 0.45, 1)
            cv2.rectangle(img_draw, (vx1, vy1 - th - 4), (vx1 + tw + 4, vy1), color_v, -1)
            cv2.putText(img_draw, label_v, (vx1 + 2, vy1 - 2), font, 0.45, (255, 255, 255), 1)

    # Encode result image to base64
    _, buffer = cv2.imencode('.jpg', img_draw)
    import base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Save to database
    for item in all_violation_data:
        for viol in item['violations']:
            insert_violation(None, viol['type'], viol['details'], None)

    return {
        "vehicles": [
            {"bbox": list(v[0]), "vtype": v[1], "color": v[2], "conf": v[3]}
            for v in vehicles
        ],
        "matched_plates": [
            {"bbox": list(m[0]), "vtype": m[1], "color": m[2], "plate_text": m[4]}
            for m in matched
        ],
        "violations": all_violation_data,
        "red_light_violations": [
            {"bbox": list(v['bbox']), "class_name": v['class_name'], "conf": v['conf'], "details": v['details']}
            for v in red_light_violations
        ],
        "image_base64": image_base64,
        "stats": {
            "total_vehicles": len(vehicles),
            "total_violations": len(all_violation_data),
            "total_red_light": len(red_light_violations),
        }
    }


@app.post("/api/process-video")
async def process_video(
    file: UploadFile = File(...),
    enable_violation_detection: bool = Form(True),
    enable_red_light_detection: bool = Form(False),
    violation_conf_limit: float = Form(0.15),
    conf_more_than_two: float = Form(0.50),
    conf_no_helmet: float = Form(0.15),
    conf_using_mobile: float = Form(0.15),
    traffic_light_conf: float = Form(0.25),
    max_frames: int = Form(100),
    speed_limit: int = Form(60),
):
    """Process video file frame by frame and return summary."""
    # Save uploaded video to temp file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(contents)
        tfile.flush()
        video_path = tfile.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_skip = max(1, total_frames // max_frames)

        # Set config
        recognizer.set_violation_conf_limit(violation_conf_limit)
        recognizer.violation_conf_more_than_two = conf_more_than_two
        recognizer.violation_conf_without_helmet = conf_no_helmet
        recognizer.violation_conf_using_mobile = conf_using_mobile
        recognizer.set_traffic_light_conf(traffic_light_conf)

        tracker = ByteTrackVehicleTracker(recognizer)
        tracker.set_speed_limit(speed_limit)
        all_detected = []
        all_violations = []
        all_speed_violations = []
        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Process frame
                vehicles = recognizer.detect_vehicles(frame)
                plates = recognizer.detect_plates(frame)
                matched = recognizer.match_plates_to_vehicles(vehicles, plates)

                vehicle_violation_map = {}
                if enable_violation_detection:
                    vehicle_violation_map = build_hierarchical_violation_map(recognizer, vehicles, image=frame)

                # Tracking
                dets = build_tracking_dets(vehicles, recognizer.vehicle_classes)
                tracked = tracker.update(frame, dets)

                for trk_id, bbox, cls_id, conf in tracked:
                    x1, y1, x2, y2 = bbox
                    found = find_vehicle_context_by_bbox(vehicles, matched, bbox)
                    if found is None:
                        continue
                    matched_bbox, vtype, color, plate_img, plate_text = found
                    vehicle_viols = vehicle_violation_map.get(tuple(matched_bbox), [])

                    # Speed violation detection
                    speed = tracker.calculate_speed(trk_id)
                    if speed is not None and speed > speed_limit:
                        speed_violation = {
                            "frame": processed_frames,
                            "track_id": trk_id,
                            "speed_kmh": speed,
                            "speed_limit": speed_limit,
                            "plate_text": plate_text,
                            "vtype": vtype,
                            "color": color,
                            "bbox": list(bbox),
                        }
                        all_speed_violations.append(speed_violation)
                        
                        # Save to database
                        vehicle_id = None
                        speed_img_path = None
                        if plate_text:
                            vehicle_crop, _ = crop_vehicle_context(frame, (x1, y1, x2, y2), vtype)
                            speed_img_path = f"data/vehicles/speed_video_{trk_id}_{int(time.time())}.jpg"
                            os.makedirs("data/vehicles", exist_ok=True)
                            cv2.imwrite(speed_img_path, vehicle_crop)
                            
                            # Try to find or create vehicle entry
                            from database_test import get_test_connection
                            conn = get_test_connection()
                            existing = conn.execute(
                                'SELECT id FROM test_vehicles WHERE track_id = ?', (trk_id,)
                            ).fetchone()
                            if existing:
                                vehicle_id = existing['id']
                            else:
                                vehicle_id = insert_vehicle_entry(trk_id, plate_text, vtype, color, None, None)
                            conn.close()
                            
                            db_insert_speed_violation(vehicle_id, trk_id, plate_text, vtype, color, speed, speed_limit, speed_img_path)

                    if plate_text or vehicle_viols:
                        all_detected.append({
                            "frame": processed_frames,
                            "track_id": trk_id,
                            "bbox": list(bbox),
                            "plate_text": plate_text,
                            "vtype": vtype,
                            "color": color,
                            "violations": [v[0] for v in vehicle_viols]
                        })

                        for vtype_v, details, _vbbox, vconf in vehicle_viols:
                            all_violations.append({
                                "frame": processed_frames,
                                "track_id": trk_id,
                                "type": vtype_v,
                                "details": details,
                                "plate_text": plate_text,
                                "conf": vconf
                            })

                processed_frames += 1
                if processed_frames >= max_frames:
                    break

            frame_count += 1

        cap.release()

        # Get last frame with detections as result image
        result_image = None
        if processed_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
            ret, last_frame = cap.read()
            if ret:
                # Draw detections on last frame
                img_draw = last_frame.copy()
                for det in all_detected[-10:]:
                    if det['bbox']:
                        x1, y1, x2, y2 = det['bbox']
                        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{det['color']} {det['vtype']} | {det['plate_text']}"
                        cv2.putText(img_draw, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                _, buffer = cv2.imencode('.jpg', img_draw)
                import base64
                result_image = base64.b64encode(buffer).decode('utf-8')

        return {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "fps": fps,
            "detected": all_detected,
            "violations": all_violations,
            "speed_violations": all_speed_violations,
            "stats": {
                "total_vehicles": len(set(d['track_id'] for d in all_detected)),
                "total_violations": len(all_violations),
                "total_speed_violations": len(all_speed_violations),
            },
            "result_image": result_image
        }

    finally:
        # Cleanup temp file
        if os.path.exists(video_path):
            os.unlink(video_path)


# ======================== GLOBAL DASHBOARD API ========================

@app.get("/api/global-stats")
def get_global_dashboard_stats():
    """Get combined stats from all databases for Dashboard."""
    return get_global_stats()


# ======================== PARKING DATABASE API ========================

@app.get("/api/parking/stats")
def get_parking_db_stats():
    return get_parking_stats()

@app.get("/api/parking/entries")
def get_parking_db_entries(limit: int = Query(100, le=500), offset: int = Query(0, ge=0)):
    return get_parking_entries(limit, offset)

@app.get("/api/parking/slots")
def get_parking_db_slots():
    return get_parking_slots()


# ======================== LOGISTICS DATABASE API ========================

@app.get("/api/logistics/stats")
def get_logistics_db_stats():
    return get_logistics_stats()

@app.get("/api/logistics/entries")
def get_logistics_db_entries(limit: int = Query(100, le=500), offset: int = Query(0, ge=0)):
    return get_logistics_entries(limit, offset)

@app.get("/api/logistics/unknown-alerts")
def get_logistics_db_unknown_alerts(limit: int = Query(50, le=200), offset: int = Query(0, ge=0)):
    return get_unknown_alerts(limit, offset)

@app.get("/api/logistics/truck-visits")
def get_logistics_db_truck_visits(limit: int = Query(50, le=200), offset: int = Query(0, ge=0)):
    return get_truck_visits()


# ======================== SMARTCITY DATABASE API ========================

@app.get("/api/smartcity/stats")
def get_smartcity_db_stats():
    return get_smartcity_stats()

@app.get("/api/smartcity/flow")
def get_smartcity_db_flow(limit: int = Query(100, le=500), offset: int = Query(0, ge=0)):
    return get_traffic_flow(limit, offset)

@app.get("/api/smartcity/violations")
def get_smartcity_db_violations(limit: int = Query(100, le=500), offset: int = Query(0, ge=0)):
    return get_city_violations(limit, offset)

@app.get("/api/smartcity/flow-by-hour")
def get_smartcity_db_flow_by_hour():
    return get_flow_by_hour()


# ======================== WEBCAM / RTSP ========================

class CameraStream:
    """Shared camera stream manager with tracking and violation detection."""
    def __init__(self):
        self.cap = None
        self.running = False
        self.source = None
        self.lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.tracker = None
        self.active_tracks = {}
        self.last_seen = {}
        self.sent_plates = set()
        self.sent_violations = set()
        self.sent_red_light_violations = set()
        self.detected_items = []
        self.detected_track_ids = set()  # Track IDs already added to detected_items (ensure ONE entry per vehicle)
        self.detected_track_index = {}   # track_id -> index in detected_items (for fast updates)
        self.violation_items = []
        self.red_light_items = []
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.unknown_vehicle_alerts = []  # For logistics unknown vehicle detection
        self._raw_frame = None           # Raw frame for immediate display
        self._last_frame = None          # Processed frame with overlays
        self._processing_frame = None    # Frame currently being processed
        self._processing_done = threading.Event()
        self._violation_keys = set()  # Deduplicate violations per track+type

    def start(self, source, backend=cv2.CAP_ANY, target_fps=20):
        self.stop()
        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self.source = source
        self.target_fps = target_fps
        self.tracker = ByteTrackVehicleTracker(recognizer)
        self.active_tracks.clear()
        self.last_seen.clear()
        self.sent_plates.clear()
        self.sent_violations.clear()
        self.sent_red_light_violations.clear()
        self.detected_items.clear()
        self.detected_track_ids.clear()  # Reset track IDs for new session
        self.violation_items.clear()
        self.red_light_items.clear()
        self.unknown_vehicle_alerts.clear()
        self._raw_frame = None
        self._last_frame = None
        self._processing_frame = None
        self._processing_done = threading.Event()
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        # Start capture loop (high FPS, non-blocking)
        threading.Thread(target=self._capture_loop, daemon=True).start()
        # Start processing loop (low FPS, heavy YOLO processing)
        threading.Thread(target=self._processing_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _capture_loop(self):
        """Capture loop: read frames at 20 FPS, always keep latest raw frame available."""
        target_fps = getattr(self, 'target_fps', 20)
        frame_interval = 1.0 / target_fps
        
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            
            now = time.time()
            
            # Store raw frame immediately for instant display
            with self.lock:
                self._raw_frame = frame.copy()
                # Draw BEV zone on raw frame if configured
                try:
                    pts = bev_detector.config.src_points.astype(np.int32)
                    if np.any(pts != 0):
                        cv2.polylines(self._raw_frame, [pts], True, (0, 0, 255), 2)
                        try:
                            left, right = bev_detector.get_stop_line_2d()
                            cv2.line(self._raw_frame, left, right, (0, 152, 255), 2)
                        except Exception:
                            pass
                except Exception:
                    pass
            
            # FPS tracking
            self.frame_count += 1
            if now - self.last_fps_time > 0.5:
                self.fps = self.frame_count / (now - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = now
            
            # Signal processing thread if it's free
            if self._processing_done.is_set() or self._processing_frame is None:
                self._processing_frame = frame
                self._processing_done.clear()
            
            # Maintain steady FPS
            sleep_time = frame_interval - (time.time() - now)
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def _processing_loop(self):
        """Processing loop: runs detection asynchronously."""
        while self.running:
            frame = None
            with self.lock:
                if self._processing_frame is not None and not self._processing_done.is_set():
                    frame = self._processing_frame
                    self._processing_done.clear()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            try:
                # Run heavy detection+drawing (modifies frame in-place, stores in _last_frame)
                self._process_frame(frame)
            except Exception as e:
                print(f"[CameraStream] Processing error: {e}")
            
            with self.lock:
                self._processing_frame = None
                self._processing_done.set()

    def _process_frame(self, frame):
        # FPS calculation
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time > 0.5:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now

        # Detection
        vehicles = recognizer.detect_vehicles(frame)
        plates = recognizer.detect_plates(frame)
        matched = recognizer.match_plates_to_vehicles(vehicles, plates)

        # Violations
        vehicle_violation_map = build_hierarchical_violation_map(recognizer, vehicles, image=frame)

        # Tracking
        dets = build_tracking_dets(vehicles, recognizer.vehicle_classes)
        tracked = self.tracker.update(frame, dets)

        # ======================== OCR CONSOLIDATION ========================
        # Feed per-frame OCR results into the consolidator for each tracked vehicle.
        # The consolidator uses voting across frames to determine the best plate text.
        # This prevents: duplicate DB entries, wrong OCR reads, and multiple records per vehicle.
        for trk_id, bbox, cls_id, conf in tracked:
            found = find_vehicle_context_by_bbox(vehicles, matched, bbox)
            if found is not None:
                matched_bbox, vtype, color, plate_img, plate_text = found
                track_mem = ocr_consolidator.get_track_memory(trk_id)
                if plate_text:
                    track_mem.add_ocr_result(plate_text)
                    # Store best crop for evidence
                    if track_mem.best_vehicle_crop is None:
                        vehicle_crop, _ = crop_vehicle_context(frame, bbox, vtype)
                        track_mem.best_vehicle_crop = vehicle_crop
                        track_mem.best_plate_img = plate_img
        # ======================== END OCR CONSOLIDATION ========================

        # Always detect traffic lights for drawing
        camera_traffic_lights, camera_traffic_road_users = recognizer.detect_traffic_scene(frame)

        # ROI-based red light violation detection
        roi_config = roi_detector.get_config()
        red_light_active = False
        if roi_config and traffic_light_model is not None:
            try:
                for tl in camera_traffic_lights:
                    tl_bbox = tl.get('bbox', [])
                    if len(tl_bbox) == 4:
                        tx1, ty1, tx2, ty2 = tl_bbox
                        # Check if traffic light is in upper portion of frame
                        light_region = frame[ty1:ty2, tx1:tx2]
                        if light_region.size > 0:
                            # Simple red light detection: check for red dominant color
                            hsv = cv2.cvtColor(light_region, cv2.COLOR_BGR2HSV)
                            lower_red1 = np.array([0, 100, 100])
                            upper_red1 = np.array([10, 255, 255])
                            lower_red2 = np.array([160, 100, 100])
                            upper_red2 = np.array([180, 255, 255])
                            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                            red_mask = cv2.bitwise_or(mask1, mask2)
                            red_ratio = np.sum(red_mask > 0) / red_mask.size
                            if red_ratio > 0.15:
                                red_light_active = True
                                break
            except Exception:
                pass

        # ROI violation detection
        roi_violations = []
        if roi_config and red_light_active:
            vehicle_list = []
            for trk_id, bbox, cls_id, conf in tracked:
                x1, y1, x2, y2 = bbox
                vehicle_list.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "track_id": trk_id,
                    "class_name": recognizer.vehicle_classes.get(cls_id, "vehicle"),
                    "conf": float(conf),
                })
            roi_violations = roi_detector.process_vehicles(vehicle_list, red_light_active=True)

        # Bird's Eye View (BEV) red light detection (like Colab)
        bev_detector.update_red_light(camera_traffic_lights)
        
        for trk_id, bbox, cls_id, conf in tracked:
            x1, y1, x2, y2 = bbox
            class_name = recognizer.vehicle_classes.get(cls_id, "vehicle")
            bev_result = bev_detector.process_vehicle(trk_id, class_name, (x1, y1, x2, y2))
            
            if bev_result["first_time_violation"]:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] 🔴 BEV: Vehicle {bev_result['unique_key'].upper()} crossed 3D boundary at red light!")
                
                # Add BEV violation to violation_items for display on frontend
                try:
                    vehicle_crop, _ = crop_vehicle_context(frame, (x1, y1, x2, y2), class_name)
                    if vehicle_crop is not None:
                        bev_violation_key = f"{trk_id}_RED_LIGHT_BEV"
                        if bev_violation_key not in self.sent_violations:
                            self.sent_violations.add(bev_violation_key)
                            
                            # Find vehicle info from matched plates and vehicle data
                            found = find_vehicle_context_by_bbox(vehicles, matched, bbox)
                            if found is not None:
                                matched_bbox, vtype, color, plate_img, plate_text = found
                            else:
                                vtype = class_name
                                color = "unknown"
                                plate_text = ""
                            
                            viol_details = f"Red light running (BEV 3D) - {vtype}"
                            
                            # Use consolidated plate from OCR consolidator
                            track_mem = ocr_consolidator.get_track_memory(trk_id)
                            consolidated_plate, _ = track_mem.get_best_plate()
                            display_plate = consolidated_plate if consolidated_plate else plate_text
                            
                            if display_plate:
                                viol_info = f"🚨 RED LIGHT {display_plate} ({color} {vtype})"
                            else:
                                viol_info = f"🚨 RED LIGHT 3D: {bev_result['unique_key'].upper()}"
                            # Store with full info: (crop, plate_text, vtype, details, viol_info, vtype, color, trk_id)
                            self.violation_items.append((vehicle_crop, display_plate, "RED_LIGHT_VIOLATION", viol_details, viol_info, vtype, color, trk_id))
                            # Also add to red_light_items for additional tracking
                            self.red_light_items.append({
                                "time": time.time(),
                                "track_id": trk_id,
                                "class_name": vtype,
                                "bbox": (x1, y1, x2, y2),
                                "details": viol_details,
                                "unique_key": bev_result["unique_key"],
                                "plate_text": plate_text,
                                "color": color,
                            })
                            
                            # Save violation to database
                            vehicle_id = self.active_tracks.get(trk_id)
                            if vehicle_id is None:
                                vehicle_img_path = f"data/vehicles/track_{trk_id}_{int(time.time())}.jpg"
                                os.makedirs("data/vehicles", exist_ok=True)
                                cv2.imwrite(vehicle_img_path, vehicle_crop)
                                vehicle_id = insert_vehicle_entry(trk_id, "", class_name, "unknown", vehicle_img_path, None)
                                self.active_tracks[trk_id] = vehicle_id
                            viol_img_path = f"data/vehicles/violation_bev_{trk_id}_{int(time.time())}.jpg"
                            cv2.imwrite(viol_img_path, vehicle_crop)
                            insert_violation(vehicle_id, "RED_LIGHT_VIOLATION", viol_details, viol_img_path)
                            # Send Telegram notification
                            threading.Thread(
                                target=send_violation_telegram,
                                args=(vehicle_crop, bev_result["unique_key"].upper(), "RED_LIGHT_VIOLATION", viol_details)
                            ).start()
                except Exception as e:
                    print(f"[CameraStream] Error handling BEV violation: {e}")

        current_displayed_plate_texts = {item[2] for item in self.detected_items}
        current_displayed_violations = {f"{item[1]}_{item[2]}" for item in self.violation_items}

        # Speed violation detection
        speed_violation_sent = set()
        for trk_id, bbox, cls_id, conf in tracked:
            speed = self.tracker.calculate_speed(trk_id)
            if speed is not None and speed > self.tracker.speed_limit:
                speed_violation_key = f"{trk_id}_SPEED"
                if speed_violation_key not in speed_violation_sent:
                    speed_violation_sent.add(speed_violation_key)
                    x1, y1, x2, y2 = bbox
                    found = find_vehicle_context_by_bbox(vehicles, matched, bbox)
                    if found is not None:
                        matched_bbox, vtype, color, plate_img, plate_text = found
                        vehicle_id = self.active_tracks.get(trk_id)
                        if vehicle_id is None:
                            vehicle_crop, _ = crop_vehicle_context(frame, (x1, y1, x2, y2), vtype)
                            vehicle_img_path = f"data/vehicles/track_{trk_id}_{int(time.time())}.jpg"
                            plate_img_path = None
                            os.makedirs("data/vehicles", exist_ok=True)
                            os.makedirs("data/plates", exist_ok=True)
                            cv2.imwrite(vehicle_img_path, vehicle_crop)
                            if plate_img is not None:
                                plate_img_path = f"data/plates/track_{trk_id}_{int(time.time())}.jpg"
                                cv2.imwrite(plate_img_path, plate_img)
                            vehicle_id = insert_vehicle_entry(trk_id, plate_text, vtype, color, vehicle_img_path, plate_img_path)
                            self.active_tracks[trk_id] = vehicle_id
                        
                        speed_img_path = f"data/vehicles/speed_{trk_id}_{int(time.time())}.jpg"
                        vehicle_crop, _ = crop_vehicle_context(frame, (x1, y1, x2, y2), vtype)
                        cv2.imwrite(speed_img_path, vehicle_crop)
                        db_insert_speed_violation(vehicle_id, trk_id, plate_text, vtype, color, speed, self.tracker.speed_limit, speed_img_path)
                        speed_details = f"Speed {speed} km/h (limit {self.tracker.speed_limit} km/h)"
                        self.violation_items.append((vehicle_crop, plate_text, "SPEED_VIOLATION", speed_details, f"{speed} km/h", vtype, color))
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"[{current_time}] ⚠️ SPEED: Vehicle {plate_text or trk_id} - {speed} km/h (limit {self.tracker.speed_limit} km/h)")

        for trk_id, bbox, cls_id, conf in tracked:
            self.last_seen[trk_id] = time.time()
            x1, y1, x2, y2 = bbox
            found = find_vehicle_context_by_bbox(vehicles, matched, bbox)
            if found is None:
                continue
            matched_bbox, vtype, color, plate_img, plate_text = found
            vehicle_viols = vehicle_violation_map.get(tuple(matched_bbox), [])
            
            # Get consolidated plate text from OCR consolidator (voting across frames)
            track_mem = ocr_consolidator.get_track_memory(trk_id)
            consolidated_plate, is_finalized = track_mem.get_best_plate()
            display_plate = consolidated_plate if consolidated_plate else plate_text
            
            # Skip if no plate and no violations
            if not display_plate and not vehicle_viols:
                continue
            
            # Check if this track is new to detected_items (not just active_tracks)
            is_new_detection = trk_id not in self.detected_track_ids
            
            # Only add to detected_items ONCE per track (when first seen)
            if is_new_detection:
                self.detected_track_ids.add(trk_id)
                vehicle_crop, _ = crop_vehicle_context(frame, (x1, y1, x2, y2), vtype)
                vehicle_img_path = f"data/vehicles/track_{trk_id}_{int(time.time())}.jpg"
                plate_img_path = None
                os.makedirs("data/vehicles", exist_ok=True)
                os.makedirs("data/plates", exist_ok=True)
                cv2.imwrite(vehicle_img_path, vehicle_crop)
                if plate_img is not None:
                    plate_img_path = f"data/plates/track_{trk_id}_{int(time.time())}.jpg"
                    cv2.imwrite(plate_img_path, plate_img)
                
                # Use consolidated plate for DB entry
                vehicle_id = insert_vehicle_entry(trk_id, display_plate, vtype, color, vehicle_img_path, plate_img_path)
                self.active_tracks[trk_id] = vehicle_id
                
                # Store best crops in consolidator
                if track_mem.best_vehicle_crop is None:
                    track_mem.best_vehicle_crop = vehicle_crop
                if track_mem.best_plate_img is None and plate_img is not None:
                    track_mem.best_plate_img = plate_img
                
                # Add to detected_items ONCE with consolidated plate
                info = f"{color} {vtype} | {display_plate}"
                self.detected_items.append((track_mem.best_vehicle_crop, track_mem.best_plate_img, display_plate, info))
                # Store index and vote count for fast updates later
                current_votes = track_mem.ocr_results.get(display_plate, 0)
                self.detected_track_index[trk_id] = (len(self.detected_items) - 1, current_votes)
                
                # Send Telegram notification for new plate
                if display_plate and display_plate not in self.sent_plates:
                    self.sent_plates.add(display_plate)
                    threading.Thread(
                        target=send_telegram_notification,
                        args=(vehicle_crop, plate_img, display_plate, vtype, color)
                    ).start()
                
                # Check for unknown vehicle - no plate detected
                if not display_plate or display_plate.strip() == "":
                    alert_info = {
                        "time": time.time(),
                        "track_id": trk_id,
                        "vehicle_type": vtype,
                        "color": color,
                        "message": f"Unknown vehicle detected: {color} {vtype} (no license plate)",
                    }
                    self.unknown_vehicle_alerts.append(alert_info)
                    if len(self.unknown_vehicle_alerts) > 50:
                        self.unknown_vehicle_alerts.pop(0)
            else:
                # Update existing detected_items entry with better plate if available
                # Only update if we have a finalized BETTER plate (not just different)
                if is_finalized and consolidated_plate:
                    # Use direct index lookup for O(1) update
                    idx_data = self.detected_track_index.get(trk_id)
                    if idx_data is not None:
                        idx, old_votes = idx_data
                        if 0 <= idx < len(self.detected_items):
                            old_crop, old_plate_img, old_plate_text, old_info = self.detected_items[idx]
                            # Only update if the new plate has MORE votes than current
                            new_votes = track_mem.ocr_results.get(consolidated_plate, 0)
                            if consolidated_plate != old_plate_text and new_votes > old_votes:
                                new_info = f"{color} {vtype} | {consolidated_plate}"
                                self.detected_items[idx] = (track_mem.best_vehicle_crop or old_crop, track_mem.best_plate_img or old_plate_img, consolidated_plate, new_info)
                                # Update stored vote count
                                self.detected_track_index[trk_id] = (idx, new_votes)

            # Handle violations (deduplicate per track+type)
            for vtype_v, details, _vbbox, vconf in vehicle_viols:
                violation_key = f"{trk_id}_{vtype_v}"
                if violation_key in self.sent_violations:
                    continue
                self.sent_violations.add(violation_key)
                
                vehicle_crop, _ = crop_vehicle_context(frame, (x1, y1, x2, y2), vtype)
                viol_img_path = f"data/vehicles/violation_{trk_id}_{int(time.time())}.jpg"
                cv2.imwrite(viol_img_path, vehicle_crop)
                
                vehicle_id = self.active_tracks.get(trk_id)
                if vehicle_id:
                    insert_violation(vehicle_id, vtype_v, details, viol_img_path)
                
                viol_info = f"{details} ({vconf*100:.1f}%)"
                # Use the best consolidated plate for violations
                self.violation_items.append((vehicle_crop, display_plate, vtype_v, details, viol_info, vtype, color))
                threading.Thread(
                    target=send_violation_telegram,
                    args=(vehicle_crop, display_plate, vtype_v, details)
                ).start()

        # Remove old tracks + cleanup OCR consolidator
        current_time = time.time()
        to_remove = []
        for trk_id, veh_id in self.active_tracks.items():
            if current_time - self.last_seen.get(trk_id, 0) > 2.0:
                update_exit_vehicle(veh_id)
                to_remove.append(trk_id)
        for trk_id in to_remove:
            del self.active_tracks[trk_id]
            if trk_id in self.last_seen:
                del self.last_seen[trk_id]
            # Remove from detected_track_ids to allow re-detection if vehicle comes back
            self.detected_track_ids.discard(trk_id)
            # Remove from index mapping
            self.detected_track_index.pop(trk_id, None)
            # Finalize plate for this track and cleanup OCR memory
            final_plate = ocr_consolidator.finalize_plate_for_track(trk_id)
            if final_plate:
                current_time_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time_str}] 🏁 Track {trk_id} exited (final plate: {final_plate})")
            ocr_consolidator.remove_track(trk_id)
        
        # Periodic OCR consolidator cleanup (removes stale track memories)
        removed = ocr_consolidator.cleanup()
        if removed > 0:
            print(f"[OCR] Cleaned up {removed} stale track memories")

        # Draw ROI zone
        if roi_config is not None:
            frame = roi_detector.draw_roi(frame)

        # Draw traffic lights
        state_colors = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 200, 0),
            "unknown": (180, 180, 180)
        }
        for light in camera_traffic_lights:
            x1, y1, x2, y2 = light["bbox"]
            color = state_colors.get(light["state"], (180, 180, 180))
            label = f"{light['state'].upper()} {light['conf']*100:.1f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw tracked vehicles with speed
        for trk_id, bbox, cls_id, conf in tracked:
            x1, y1, x2, y2 = bbox
            color_box = (0, 255, 0) if trk_id in self.active_tracks else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 1)
            # Speed display
            speed = self.tracker.calculate_speed(trk_id)
            speed_text = f"ID:{trk_id}"
            if speed is not None:
                speed_text += f" {speed:.0f}km/h"
            cv2.putText(frame, speed_text, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_box, 1)
        
        # Draw license plate bounding boxes
        for plate_img, (px1, py1, px2, py2) in plates:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
            cv2.putText(frame, "PLATE", (px1, py1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw matched plate text near vehicles
        for m in matched:
            if len(m) >= 5:
                mb, mt, mc, mp_img, mp_txt = m
                mx1, my1, mx2, my2 = mb
                if mp_txt:
                    cv2.putText(frame, mp_txt, (mx1, my2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Bird's Eye View (BEV) - draw all overlays like Colab
        # Draw: yellow trapezoid, camera stop line, VIOLATION/WAITING labels, light status
        frame = bev_detector.draw_all(frame, tracked, recognizer.vehicle_classes)

        # Draw ROI violations
        for viol in roi_violations:
            vx1, vy1, vx2, vy2 = viol['bbox']
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
            label = f"RED LIGHT RUNNING (ROI) {viol['conf']*100:.1f}%"
            cv2.putText(frame, label, (vx1, vy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Red light status indicator
        if roi_config is not None:
            light_color = (0, 0, 255) if red_light_active else (0, 255, 0)
            light_label = "RED LIGHT" if red_light_active else "GREEN LIGHT"
            cv2.circle(frame, (30, 60), 10, light_color, -1)
            cv2.putText(frame, light_label, (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color, 2)

        # FPS overlay
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Keep last frame for streaming
        self._last_frame = frame

    def get_last_frame(self):
        with self.lock:
            # Return processed frame which has ALL overlays: BEV zone, vehicles, violations
            # _last_frame contains full detection results (violations, tracking, etc.)
            if self._last_frame is not None:
                return self._last_frame
            # Fallback to raw frame if processed frame not ready yet
            return self._raw_frame


# ======================== VIDEO STREAMING ========================

class VideoStreamProcessor:
    """Process uploaded video and stream with bounding boxes."""
    def __init__(self):
        self.cap = None
        self.running = False
        self.paused = False
        self.tracker = None
        self.config = {}
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self._last_frame = None
        self.lock = threading.Lock()
        self.detected_items = []
        self.detected_track_ids = set()  # Track IDs already added to detected_items (ensure ONE entry per vehicle)
        self.detected_track_index = {}   # track_id -> (index, vote_count) in detected_items (for fast updates)
        self.violation_items = []
        self._violation_keys = set()  # Track IDs of violations already recorded (per type)
        self._paused_frame = None      # Store last frame when paused
        self._video_ended = False      # Flag for when video reaches end

    def start(self, video_path, config):
        self.stop()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video")
        self.running = True
        self.paused = False
        self._video_ended = False
        self.config = config
        self.tracker = ByteTrackVehicleTracker(recognizer)
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.detected_items = []
        self.detected_track_ids.clear()  # Reset track IDs for new video
        self.detected_track_index.clear()  # Reset index mapping
        self.violation_items = []
        self._violation_keys.clear()
        self._paused_frame = None
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.paused = False
        # Don't release cap here - it causes threading issues on Windows
        # Let it be released in the loop thread or garbage collected
        self.cap = None

    def pause(self):
        """Pause the video stream - freeze on current frame."""
        if self.running and not self.paused:
            self.paused = True
            print("[VideoStream] Paused")

    def resume(self):
        """Resume the video stream from current position."""
        if self.running and self.paused:
            self.paused = False
            print("[VideoStream] Resumed")

    def _loop(self):
        target_fps = getattr(self, 'target_fps', 20)
        frame_interval = 1.0 / target_fps
        process_interval = 1.0 / min(target_fps, 10)  # Process detection at max 10 FPS
        last_frame_time = 0
        last_process_time = 0
        paused_frame_sent = False
        
        while self.running and self.cap and self.cap.isOpened():
            # If paused, just keep the last frame and sleep
            if self.paused:
                if not paused_frame_sent and self._last_frame is not None:
                    with self.lock:
                        self._last_frame = self._paused_frame if self._paused_frame is not None else self._last_frame
                    paused_frame_sent = True
                time.sleep(0.05)
                continue
            paused_frame_sent = False
            
            try:
                ret, frame = self.cap.read()
            except Exception as e:
                print(f"[VideoStream] Read error (video likely ended): {e}")
                ret = False
                frame = None
            
            if not ret:
                # Video ended - auto-stop
                print("[VideoStream] Video ended, auto-stopping...")
                self._video_ended = True
                self.running = False
                # Release cap in this thread to avoid threading issues
                try:
                    if self.cap:
                        self.cap.release()
                except Exception as e:
                    print(f"[VideoStream] Release error (non-critical): {e}")
                self.cap = None
                # Keep the last frame so frontend can still display it
                break
            
            now = time.time()
            # Cap display at target_fps (20 FPS)
            if now - last_frame_time < frame_interval:
                continue
            last_frame_time = now
            
            # Only run detection at reduced rate
            if now - last_process_time >= process_interval:
                self._process_frame(frame)
                # Store a copy of this frame for pause state
                with self.lock:
                    self._paused_frame = self._last_frame.copy() if self._last_frame is not None else frame.copy()
                last_process_time = now
                continue  # Already have frame from process_frame
            
            # On skipped frames, use last processed frame
            with self.lock:
                frame_to_display = self._last_frame if self._last_frame is not None else frame
            
            if frame_to_display is not None:
                self._last_frame = frame_to_display
                
            # Maintain steady FPS
            sleep_time = frame_interval - (time.time() - now)
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def _process_frame(self, frame):
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time > 0.5:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now

        # Detection
        vehicles = recognizer.detect_vehicles(frame)
        plates = recognizer.detect_plates(frame)
        matched = recognizer.match_plates_to_vehicles(vehicles, plates)

        # Violations
        vehicle_violation_map = {}
        if self.config.get('enable_violation_detection', True):
            vehicle_violation_map = build_hierarchical_violation_map(recognizer, vehicles, image=frame)

        # Tracking
        dets = build_tracking_dets(vehicles, recognizer.vehicle_classes)
        tracked = self.tracker.update(frame, dets)

        # ======================== OCR CONSOLIDATION (Video Stream) ========================
        for trk_id, bbox, cls_id, conf in tracked:
            found = find_vehicle_context_by_bbox(vehicles, matched, bbox)
            if found is not None:
                matched_bbox, vtype, color, plate_img, plate_text = found
                track_mem = ocr_consolidator.get_track_memory(trk_id)
                if plate_text:
                    track_mem.add_ocr_result(plate_text)
                    if track_mem.best_vehicle_crop is None:
                        vehicle_crop, _ = crop_vehicle_context(frame, bbox, vtype)
                        track_mem.best_vehicle_crop = vehicle_crop
                        track_mem.best_plate_img = plate_img
        # ======================== END OCR CONSOLIDATION ========================

        # ROI-based red light violation detection
        roi_config = roi_detector.get_config()
        red_light_active = False
        if roi_config and traffic_light_model is not None:
            try:
                traffic_lights, _ = recognizer.detect_traffic_scene(frame)
                for tl in traffic_lights:
                    tl_bbox = tl.get('bbox', [])
                    if len(tl_bbox) == 4:
                        tx1, ty1, tx2, ty2 = tl_bbox
                        light_region = frame[ty1:ty2, tx1:tx2]
                        if light_region.size > 0:
                            hsv = cv2.cvtColor(light_region, cv2.COLOR_BGR2HSV)
                            lower_red1 = np.array([0, 100, 100])
                            upper_red1 = np.array([10, 255, 255])
                            lower_red2 = np.array([160, 100, 100])
                            upper_red2 = np.array([180, 255, 255])
                            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                            red_mask = cv2.bitwise_or(mask1, mask2)
                            red_ratio = np.sum(red_mask > 0) / red_mask.size
                            if red_ratio > 0.15:
                                red_light_active = True
                                break
            except Exception:
                pass

        # ROI violation detection
        roi_violations = []
        if roi_config and red_light_active:
            vehicle_list = []
            for trk_id, bbox, cls_id, conf in tracked:
                x1, y1, x2, y2 = bbox
                vehicle_list.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "track_id": trk_id,
                    "class_name": recognizer.vehicle_classes.get(cls_id, "vehicle"),
                    "conf": float(conf),
                })
            roi_violations = roi_detector.process_vehicles(vehicle_list, red_light_active=True)

        # Always detect traffic lights for drawing
        video_traffic_lights, video_traffic_road_users = recognizer.detect_traffic_scene(frame)

        # Bird's Eye View (BEV) red light detection for video stream (like Colab)
        bev_detector.update_red_light(video_traffic_lights)
        
        for trk_id, bbox, cls_id, conf in tracked:
            x1, y1, x2, y2 = bbox
            class_name = recognizer.vehicle_classes.get(cls_id, "vehicle")
            bev_result = bev_detector.process_vehicle(trk_id, class_name, (x1, y1, x2, y2))
            
            if bev_result["first_time_violation"]:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] 🔴 BEV (Video): Vehicle {bev_result['unique_key'].upper()} crossed 3D boundary at red light!")
                
                # Add BEV violation to violation_items for display on frontend
                try:
                    vehicle_crop, _ = crop_vehicle_context(frame, (x1, y1, x2, y2), class_name)
                    if vehicle_crop is not None:
                        # Find vehicle info
                        found = find_vehicle_context_by_bbox(vehicles, matched, bbox)
                        if found is not None:
                            matched_bbox, vtype, color, plate_img, plate_text = found
                        else:
                            vtype = class_name
                            color = "unknown"
                            plate_text = ""
                        
                        # Use consolidated plate from OCR consolidator
                        track_mem = ocr_consolidator.get_track_memory(trk_id)
                        consolidated_plate, _ = track_mem.get_best_plate()
                        display_plate = consolidated_plate if consolidated_plate else plate_text
                        
                        viol_details = f"Red light running (BEV 3D) - {vtype}"
                        if display_plate:
                            viol_info = f"🚨 RED LIGHT {display_plate} ({color} {vtype})"
                        else:
                            viol_info = f"🚨 RED LIGHT 3D: {bev_result['unique_key'].upper()}"
                        self.violation_items.append((vehicle_crop, display_plate, "RED_LIGHT_VIOLATION", viol_details, viol_info, vtype, color))
                except Exception as e:
                    print(f"[VideoStream] Error handling BEV violation: {e}")

        # ======================== DEDUPLICATED DETECTED ITEMS ========================
        # Use consolidated plate text and ensure each track appears only once
        for trk_id, bbox, cls_id, conf in tracked:
            x1, y1, x2, y2 = bbox
            found = find_vehicle_context_by_bbox(vehicles, matched, bbox)
            if found is None:
                continue
            matched_bbox, vtype, color, plate_img, plate_text = found
            vehicle_viols = vehicle_violation_map.get(tuple(matched_bbox), [])
            
            # Get consolidated plate text from OCR consolidator (voting across frames)
            track_mem = ocr_consolidator.get_track_memory(trk_id)
            consolidated_plate, is_finalized = track_mem.get_best_plate()
            display_plate = consolidated_plate if consolidated_plate else plate_text
            
            # Skip if no plate and no violations
            if not display_plate and not vehicle_viols:
                continue
            
            # Check if this track is new to detected_items
            is_new_detection = trk_id not in self.detected_track_ids
            
            if is_new_detection:
                self.detected_track_ids.add(trk_id)
                vehicle_crop, _ = crop_vehicle_context(frame, (x1, y1, x2, y2), vtype)
                info = f"{color} {vtype} | {display_plate}"
                self.detected_items.append((vehicle_crop, plate_img, display_plate, info))
                # Store index and vote count for fast updates later
                current_votes = track_mem.ocr_results.get(display_plate, 0)
                self.detected_track_index[trk_id] = (len(self.detected_items) - 1, current_votes)
            else:
                # Update existing entry with better plate if available
                # Only update if new plate has MORE votes than current
                if is_finalized and consolidated_plate:
                    idx_data = self.detected_track_index.get(trk_id)
                    if idx_data is not None:
                        idx, old_votes = idx_data
                        if 0 <= idx < len(self.detected_items):
                            old_crop, old_plate_img, old_plate_text, old_info = self.detected_items[idx]
                            new_votes = track_mem.ocr_results.get(consolidated_plate, 0)
                            # Only update if new plate has more votes (higher confidence)
                            if consolidated_plate != old_plate_text and new_votes > old_votes:
                                new_info = f"{color} {vtype} | {consolidated_plate}"
                                self.detected_items[idx] = (track_mem.best_vehicle_crop or old_crop, plate_img, consolidated_plate, new_info)
                                # Update stored vote count
                                self.detected_track_index[trk_id] = (idx, new_votes)
        
        # Limit memory usage - keep only last 50 items
        if len(self.detected_items) > 50:
            self.detected_items = self.detected_items[-50:]
        if len(self.violation_items) > 50:
            self.violation_items = self.violation_items[-50:]

        # Draw
        img_draw = frame.copy()

        # Bird's Eye View (BEV) - draw all overlays like Colab
        # Draw: yellow trapezoid, camera stop line, VIOLATION/WAITING labels, light status
        # (bev_detector.update_red_light has been called above)
        img_draw = bev_detector.draw_all(img_draw, tracked, recognizer.vehicle_classes)

        # Draw ROI zone
        if roi_config is not None:
            img_draw = roi_detector.draw_roi(img_draw)

        # Draw traffic lights
        state_colors = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 200, 0),
            "unknown": (180, 180, 180)
        }
        for light in video_traffic_lights:
            x1, y1, x2, y2 = light["bbox"]
            color = state_colors.get(light["state"], (180, 180, 180))
            label = f"{light['state'].upper()} {light['conf']*100:.1f}%"
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_draw, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw tracked vehicles with speed
        for trk_id, bbox, cls_id, conf in tracked:
            x1, y1, x2, y2 = bbox
            found = find_vehicle_context_by_bbox(vehicles, matched, bbox)
            if found is None:
                continue
            matched_bbox, vtype, color, plate_img, plate_text = found
            vehicle_viols = vehicle_violation_map.get(tuple(matched_bbox), [])
            has_violation = len(vehicle_viols) > 0
            box_color = (0, 0, 255) if has_violation else (0, 255, 0)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), box_color, 2)
            # Speed display
            speed = self.tracker.calculate_speed(trk_id)
            label = f"ID:{trk_id} {color} {vtype}"
            if speed is not None:
                label += f" {speed:.0f}km/h"
            if plate_text:
                label += f" | {plate_text}"
            cv2.putText(img_draw, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Draw license plate bounding boxes
        for plate_img, (px1, py1, px2, py2) in plates:
            cv2.rectangle(img_draw, (px1, py1), (px2, py2), (255, 255, 0), 2)
            cv2.putText(img_draw, "PLATE", (px1, py1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw matched plate text near vehicles
        for m in matched:
            if len(m) >= 5:
                mb, mt, mc, mp_img, mp_txt = m
                mx1, my1, mx2, my2 = mb
                if mp_txt:
                    cv2.putText(img_draw, mp_txt, (mx1, my2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw violations
        if self.config.get('enable_violation_detection', True):
            for v in vehicles:
                v_bbox = v[0]
                if v_bbox in vehicle_violation_map:
                    vx1, vy1, vx2, vy2 = v_bbox
                    for vtype_v, details, (vvx1, vvy1, vvx2, vvy2), vconf in vehicle_violation_map[v_bbox]:
                        color_v = recognizer.get_violation_color(vtype_v)
                        cv2.rectangle(img_draw, (vvx1, vvy1), (vvx2, vvy2), color_v, 2)
                        label_v = f"{vtype_v} {vconf*100:.1f}%"
                        cv2.putText(img_draw, label_v, (vvx1, vvy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_v, 2)

        # Draw ROI violations
        for viol in roi_violations:
            vx1, vy1, vx2, vy2 = viol['bbox']
            cv2.rectangle(img_draw, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
            label = f"RED LIGHT RUNNING (ROI) {viol['conf']*100:.1f}%"
            cv2.putText(img_draw, label, (vx1, vy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Red light status indicator
        if roi_config is not None:
            light_color = (0, 0, 255) if red_light_active else (0, 255, 0)
            light_label = "RED LIGHT" if red_light_active else "GREEN LIGHT"
            cv2.circle(img_draw, (30, 60), 10, light_color, -1)
            cv2.putText(img_draw, light_label, (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color, 2)

        # FPS
        cv2.putText(img_draw, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        with self.lock:
            self._last_frame = img_draw

    def get_last_frame(self):
        with self.lock:
            # If video ended, still return the last frame (don't return None)
            if self._last_frame is not None:
                return self._last_frame
            # If video ended and we have a paused frame, return that
            if self._paused_frame is not None:
                return self._paused_frame
            return None


video_stream = VideoStreamProcessor()

@app.post("/api/video/stream")
async def start_video_stream(
    file: UploadFile = File(...),
    enable_violation_detection: bool = Form(True),
    enable_red_light_detection: bool = Form(False),
    violation_conf_limit: float = Form(0.15),
    conf_more_than_two: float = Form(0.50),
    conf_no_helmet: float = Form(0.15),
    conf_using_mobile: float = Form(0.15),
    traffic_light_conf: float = Form(0.25),
):
    """Upload video and start streaming with bounding boxes."""
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(contents)
        tfile.flush()
        video_path = tfile.name

    config = {
        'enable_violation_detection': enable_violation_detection,
        'enable_red_light_detection': enable_red_light_detection,
        'violation_conf_limit': violation_conf_limit,
        'conf_more_than_two': conf_more_than_two,
        'conf_no_helmet': conf_no_helmet,
        'conf_using_mobile': conf_using_mobile,
        'traffic_light_conf': traffic_light_conf,
    }

    # Set config
    recognizer.set_violation_conf_limit(violation_conf_limit)
    recognizer.violation_conf_more_than_two = conf_more_than_two
    recognizer.violation_conf_without_helmet = conf_no_helmet
    recognizer.violation_conf_using_mobile = conf_using_mobile
    recognizer.set_traffic_light_conf(traffic_light_conf)

    try:
        video_stream.start(video_path, config)
        return {"status": "started", "video_path": video_path}
    except Exception as e:
        if os.path.exists(video_path):
            os.unlink(video_path)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/video/stop")
async def stop_video_stream():
    video_stream.stop()
    return {"status": "stopped"}

@app.post("/api/video/pause")
async def pause_video_stream():
    video_stream.pause()
    return {"status": "paused"}

@app.post("/api/video/resume")
async def resume_video_stream():
    video_stream.resume()
    return {"status": "resumed"}

@app.get("/api/video/frame")
async def get_video_frame():
    try:
        frame = video_stream.get_last_frame()
        if frame is None:
            # Return empty response instead of 404 to avoid frontend polling errors
            return Response(content=b"", media_type="image/jpeg")
        _, buffer = cv2.imencode('.jpg', frame)
        return StreamingResponse(
            iter([buffer.tobytes()]),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        print(f"Error getting video frame: {e}")
        return Response(content=b"", media_type="image/jpeg")

@app.get("/api/video/status")
def video_stream_status():
    try:
        return {
            "running": video_stream.running,
            "paused": getattr(video_stream, 'paused', False),
            "fps": round(video_stream.fps, 1),
            "video_ended": getattr(video_stream, '_video_ended', False),
        }
    except Exception as e:
        print(f"Error getting video status: {e}")
        return {"running": False, "paused": False, "fps": 0.0, "video_ended": False}

@app.get("/api/video/detected")
def get_video_detected():
    try:
        import base64
        vehicles = []
        for p in video_stream.detected_items[-10:]:
            vehicle_crop, plate_img, plate_text, info = p
            vehicle_b64 = None
            plate_b64 = None
            if vehicle_crop is not None and vehicle_crop.size > 0:
                _, buf = cv2.imencode('.jpg', vehicle_crop)
                vehicle_b64 = base64.b64encode(buf).decode('utf-8')
            if plate_img is not None and plate_img.size > 0:
                _, buf = cv2.imencode('.jpg', plate_img)
                plate_b64 = base64.b64encode(buf).decode('utf-8')
            vehicles.append({
                "vehicle_b64": vehicle_b64,
                "plate_b64": plate_b64,
                "plate_text": plate_text,
                "info": info
            })

        violations = []
        # Deduplicate violations: keep only the latest (highest confidence) per (track_id, violation_type)
        # This prevents showing the same violation multiple times for the same vehicle
        # But different violation types for the same vehicle will still show separately
        seen_violations = {}  # key: (track_id, vtype_code) -> violation dict
        for v in reversed(video_stream.violation_items[-50:]):  # Check last 50 to find best confidence
            vehicle_crop = v[0]
            plate_text = v[1] if len(v) > 1 else ""
            vtype_code = v[2] if len(v) > 2 else "VIOLATION"
            details = v[3] if len(v) > 3 else ""
            viol_info = v[4] if len(v) > 4 else ""
            vehicle_type = v[5] if len(v) > 5 else ""
            color = v[6] if len(v) > 6 else ""
            trk_id = v[7] if len(v) > 7 else None  # track_id for deduplication
            
            # Parse vehicle_type and color from viol_info if not provided separately
            if not vehicle_type and not color:
                import re
                match = re.search(r'\((\w+)\s+(\w+)\)', viol_info)
                if match:
                    color = match.group(1)
                    vehicle_type = match.group(2)
            
            # Create unique key for this violation (track_id + violation type)
            # If no track_id, use plate_text + violation_type as key
            if trk_id is not None:
                violation_key = (trk_id, vtype_code)
            else:
                violation_key = (plate_text, vtype_code)
            
            # Only keep the first occurrence (which is the latest due to reversed iteration)
            # This ensures we keep the most recent/best confidence violation
            if violation_key not in seen_violations:
                crop_b64 = None
                if vehicle_crop is not None and vehicle_crop.size > 0:
                    _, buf = cv2.imencode('.jpg', vehicle_crop)
                    crop_b64 = base64.b64encode(buf).decode('utf-8')
                seen_violations[violation_key] = {
                    "crop_b64": crop_b64,
                    "plate_text": plate_text,
                    "type": vtype_code,
                    "details": details,
                    "info": viol_info,
                    "vehicle_type": vehicle_type,
                    "color": color,
                }
        
        # Convert to list and take only the last 10 (most recent)
        violations = list(seen_violations.values())[-10:]
        
        # Deduplicate vehicles: keep only the latest entry for each track_id
        # This prevents showing the same vehicle multiple times
        seen_vehicles = {}  # key: track_id or plate_text -> vehicle dict
        for p in reversed(video_stream.detected_items[-20:]):  # Check last 20 to find latest
            vehicle_crop, plate_img, plate_text, info = p
            # Use plate_text as key for deduplication
            vehicle_key = plate_text.strip() if plate_text and plate_text.strip() else info
            # Only keep the first occurrence (which is the latest due to reversed iteration)
            if vehicle_key not in seen_vehicles:
                vehicle_b64 = None
                plate_b64 = None
                if vehicle_crop is not None and vehicle_crop.size > 0:
                    _, buf = cv2.imencode('.jpg', vehicle_crop)
                    vehicle_b64 = base64.b64encode(buf).decode('utf-8')
                if plate_img is not None and plate_img.size > 0:
                    _, buf = cv2.imencode('.jpg', plate_img)
                    plate_b64 = base64.b64encode(buf).decode('utf-8')
                seen_vehicles[vehicle_key] = {
                    "vehicle_b64": vehicle_b64,
                    "plate_b64": plate_b64,
                    "plate_text": plate_text,
                    "info": info
                }
        
        # Convert to list and take only the last 10 (most recent)
        vehicles = list(seen_vehicles.values())[-10:]
        return {"vehicles": vehicles, "violations": violations}
    except Exception as e:
        print(f"Error getting video detected: {e}")
        import traceback
        traceback.print_exc()
        return {"vehicles": [], "violations": []}



# ======================== PARKING ENTRY / EXIT STREAMS ========================

parking_streams = {
    "entry": CameraStream(),
    "exit": CameraStream(),
}

def get_parking_stream(gate: str) -> CameraStream:
    stream = parking_streams.get(gate)
    if stream is None:
        raise HTTPException(status_code=404, detail="Invalid parking gate")
    return stream

@app.post("/api/parking/{gate}/start")
async def start_parking_gate(
    gate: str,
    source: str = Form("0"),
    backend: str = Form("AUTO")
):
    """Start entry/exit parking camera stream."""
    stream = get_parking_stream(gate)
    try:
        src = int(source) if source.isdigit() else source
        backend_map = {
            "AUTO": cv2.CAP_ANY,
            "FFMPEG": cv2.CAP_FFMPEG,
            "GSTREAMER": cv2.CAP_GSTREAMER,
        }
        backend_flag = backend_map.get(backend, cv2.CAP_ANY)
        stream.start(src, backend=backend_flag, target_fps=20)
        return {"status": "started", "gate": gate, "source": source}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/parking/{gate}/stop")
async def stop_parking_gate(gate: str):
    """Stop entry/exit parking camera stream."""
    stream = get_parking_stream(gate)
    stream.stop()
    return {"status": "stopped", "gate": gate}

@app.get("/api/parking/{gate}/frame")
async def get_parking_gate_frame(gate: str):
    """Get latest parking gate frame as JPEG."""
    stream = get_parking_stream(gate)
    frame = stream.get_last_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/api/parking/{gate}/status")
def get_parking_gate_status(gate: str):
    """Get parking gate stream status."""
    stream = get_parking_stream(gate)
    latest_plate = ""
    latest_info = ""
    if stream.detected_items:
        latest = stream.detected_items[-1]
        latest_plate = latest[2]
        latest_info = latest[3]
    return {
        "gate": gate,
        "running": stream.running,
        "source": str(stream.source) if stream.source is not None else None,
        "fps": round(stream.fps, 1),
        "detected": len(stream.detected_items),
        "violations": len(stream.violation_items),
        "latest_plate": latest_plate,
        "latest_info": latest_info,
    }

@app.get("/api/parking/{gate}/detected")
def get_parking_gate_detected(gate: str):
    """Get latest detected vehicles for a parking gate."""
    import base64
    stream = get_parking_stream(gate)
    vehicles = []
    for p in stream.detected_items[-10:]:
        vehicle_crop, plate_img, plate_text, info = p
        vehicle_b64 = None
        plate_b64 = None
        if vehicle_crop is not None and vehicle_crop.size > 0:
            _, buf = cv2.imencode('.jpg', vehicle_crop)
            vehicle_b64 = base64.b64encode(buf).decode('utf-8')
        if plate_img is not None and plate_img.size > 0:
            _, buf = cv2.imencode('.jpg', plate_img)
            plate_b64 = base64.b64encode(buf).decode('utf-8')
        vehicles.append({
            "vehicle_b64": vehicle_b64,
            "plate_b64": plate_b64,
            "plate_text": plate_text,
            "info": info,
        })
    latest = vehicles[-1] if vehicles else None
    return {"gate": gate, "vehicles": vehicles, "latest": latest}

camera_stream = CameraStream()

@app.post("/api/webcam/start")
async def start_webcam(
    source: str = Form("0"),
    backend: str = Form("AUTO")
):
    """Start webcam/RTSP stream."""
    try:
        src = int(source) if source.isdigit() else source
        backend_map = {
            "AUTO": cv2.CAP_ANY,
            "FFMPEG": cv2.CAP_FFMPEG,
            "GSTREAMER": cv2.CAP_GSTREAMER,
        }
        backend_flag = backend_map.get(backend, cv2.CAP_ANY)
        camera_stream.start(src, backend=backend_flag, target_fps=20)
        return {"status": "started", "source": source}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/webcam/stop")
async def stop_webcam():
    """Stop webcam/RTSP stream."""
    camera_stream.stop()
    return {"status": "stopped"}

@app.get("/api/webcam/frame")
async def get_webcam_frame():
    """Get latest frame from webcam as JPEG."""
    frame = camera_stream.get_last_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/api/webcam/status")
def webcam_status():
    return {
        "running": camera_stream.running,
        "source": camera_stream.source,
        "fps": round(camera_stream.fps, 1),
        "detected": len(camera_stream.detected_items),
        "violations": len(camera_stream.violation_items),
    }

@app.get("/api/webcam/detected")
def get_detected():
    import base64
    vehicles = []
    for p in camera_stream.detected_items[-20:]:
        vehicle_crop, plate_img, plate_text, info = p
        vehicle_b64 = None
        plate_b64 = None
        if vehicle_crop is not None and vehicle_crop.size > 0:
            _, buf = cv2.imencode('.jpg', vehicle_crop)
            vehicle_b64 = base64.b64encode(buf).decode('utf-8')
        if plate_img is not None and plate_img.size > 0:
            _, buf = cv2.imencode('.jpg', plate_img)
            plate_b64 = base64.b64encode(buf).decode('utf-8')
        vehicles.append({
            "vehicle_b64": vehicle_b64,
            "plate_b64": plate_b64,
            "plate_text": plate_text,
            "info": info
        })

    violations = []
    for v in camera_stream.violation_items[-20:]:
        vehicle_crop = v[0]
        plate_text = v[1] if len(v) > 1 else ""
        vtype = v[2] if len(v) > 2 else "VIOLATION"
        details = v[3] if len(v) > 3 else ""
        viol_info = v[4] if len(v) > 4 else ""
        vehicle_type = v[5] if len(v) > 5 else ""
        color = v[6] if len(v) > 6 else ""
        
        # Parse vehicle_type and color from viol_info if not provided separately
        if not vehicle_type and not color:
            import re
            match = re.search(r'\((\w+)\s+(\w+)\)', viol_info)
            if match:
                color = match.group(1)
                vehicle_type = match.group(2)
        
        crop_b64 = None
        if vehicle_crop is not None and vehicle_crop.size > 0:
            _, buf = cv2.imencode('.jpg', vehicle_crop)
            crop_b64 = base64.b64encode(buf).decode('utf-8')
        violations.append({
            "crop_b64": crop_b64,
            "plate_text": plate_text,
            "type": vtype,
            "details": details,
            "info": viol_info,
            "vehicle_type": vehicle_type,
            "color": color,
        })
    return {"vehicles": vehicles, "violations": violations}


# ======================== LOGISTICS CAMERAS ========================
# 2 cameras: gate (entry/exit) and construction_site (construction site)

logistics_streams = {
    "gate": CameraStream(),
    "construction_site": CameraStream(),
}

def get_logistics_stream(camera: str) -> CameraStream:
    stream = logistics_streams.get(camera)
    if stream is None:
        raise HTTPException(status_code=404, detail="Invalid logistics camera")
    return stream

@app.post("/api/logistics/{camera}/start")
async def start_logistics_camera(
    camera: str,
    source: str = Form("0"),
    backend: str = Form("AUTO")
):
    """Start logistics camera stream (gate or construction_site)."""
    stream = get_logistics_stream(camera)
    try:
        src = int(source) if source.isdigit() else source
        backend_map = {
            "AUTO": cv2.CAP_ANY,
            "FFMPEG": cv2.CAP_FFMPEG,
            "GSTREAMER": cv2.CAP_GSTREAMER,
        }
        backend_flag = backend_map.get(backend, cv2.CAP_ANY)
        stream.start(src, backend=backend_flag, target_fps=20)
        return {"status": "started", "camera": camera, "source": source}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/logistics/{camera}/stop")
async def stop_logistics_camera(camera: str):
    """Stop logistics camera stream."""
    stream = get_logistics_stream(camera)
    stream.stop()
    return {"status": "stopped", "camera": camera}

@app.get("/api/logistics/{camera}/frame")
async def get_logistics_camera_frame(camera: str):
    """Get latest logistics camera frame as JPEG."""
    stream = get_logistics_stream(camera)
    frame = stream.get_last_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/api/logistics/{camera}/status")
def get_logistics_camera_status(camera: str):
    """Get logistics camera stream status."""
    stream = get_logistics_stream(camera)
    latest_plate = ""
    latest_info = ""
    if stream.detected_items:
        latest = stream.detected_items[-1]
        latest_plate = latest[2]
        latest_info = latest[3]
    return {
        "camera": camera,
        "running": stream.running,
        "source": str(stream.source) if stream.source is not None else None,
        "fps": round(stream.fps, 1),
        "detected": len(stream.detected_items),
        "violations": len(stream.violation_items),
        "unknown_vehicle_alerts": len(stream.unknown_vehicle_alerts),
        "latest_plate": latest_plate,
        "latest_info": latest_info,
    }

@app.get("/api/logistics/{camera}/detected")
def get_logistics_camera_detected(camera: str):
    """Get latest detected vehicles for a logistics camera."""
    import base64
    stream = get_logistics_stream(camera)
    vehicles = []
    for p in stream.detected_items[-10:]:
        vehicle_crop, plate_img, plate_text, info = p
        vehicle_b64 = None
        plate_b64 = None
        if vehicle_crop is not None and vehicle_crop.size > 0:
            _, buf = cv2.imencode('.jpg', vehicle_crop)
            vehicle_b64 = base64.b64encode(buf).decode('utf-8')
        if plate_img is not None and plate_img.size > 0:
            _, buf = cv2.imencode('.jpg', plate_img)
            plate_b64 = base64.b64encode(buf).decode('utf-8')
        vehicles.append({
            "vehicle_b64": vehicle_b64,
            "plate_b64": plate_b64,
            "plate_text": plate_text,
            "info": info,
        })
    latest = vehicles[-1] if vehicles else None
    
    # Include unknown vehicle alerts
    unknown_alerts = []
    for alert in stream.unknown_vehicle_alerts[-20:]:
        unknown_alerts.append({
            "time": alert["time"],
            "track_id": alert["track_id"],
            "vehicle_type": alert["vehicle_type"],
            "color": alert["color"],
            "message": alert["message"],
        })
    
    return {
        "camera": camera,
        "vehicles": vehicles,
        "latest": latest,
        "unknown_vehicle_alerts": unknown_alerts,
    }


# ======================== SMART CITY CAMERAS ========================
# 4 cameras for urban traffic monitoring

smartcity_streams = {
    "cam1": CameraStream(),
    "cam2": CameraStream(),
    "cam3": CameraStream(),
    "cam4": CameraStream(),
}

def get_smartcity_stream(camera: str) -> CameraStream:
    stream = smartcity_streams.get(camera)
    if stream is None:
        raise HTTPException(status_code=404, detail="Invalid smart city camera")
    return stream

@app.post("/api/smartcity/{camera}/start")
async def start_smartcity_camera(
    camera: str,
    source: str = Form("0"),
    backend: str = Form("AUTO")
):
    """Start smart city camera stream (cam1-cam4)."""
    stream = get_smartcity_stream(camera)
    try:
        src = int(source) if source.isdigit() else source
        backend_map = {
            "AUTO": cv2.CAP_ANY,
            "FFMPEG": cv2.CAP_FFMPEG,
            "GSTREAMER": cv2.CAP_GSTREAMER,
        }
        backend_flag = backend_map.get(backend, cv2.CAP_ANY)
        stream.start(src, backend=backend_flag)
        return {"status": "started", "camera": camera, "source": source}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/smartcity/{camera}/stop")
async def stop_smartcity_camera(camera: str):
    """Stop smart city camera stream."""
    stream = get_smartcity_stream(camera)
    stream.stop()
    return {"status": "stopped", "camera": camera}

@app.get("/api/smartcity/{camera}/frame")
async def get_smartcity_camera_frame(camera: str):
    """Get latest smart city camera frame as JPEG."""
    stream = get_smartcity_stream(camera)
    frame = stream.get_last_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/api/smartcity/{camera}/status")
def get_smartcity_camera_status(camera: str):
    """Get smart city camera stream status."""
    stream = get_smartcity_stream(camera)
    latest_plate = ""
    latest_info = ""
    if stream.detected_items:
        latest = stream.detected_items[-1]
        latest_plate = latest[2]
        latest_info = latest[3]
    return {
        "camera": camera,
        "running": stream.running,
        "source": str(stream.source) if stream.source is not None else None,
        "fps": round(stream.fps, 1),
        "detected": len(stream.detected_items),
        "violations": len(stream.violation_items),
        "latest_plate": latest_plate,
        "latest_info": latest_info,
    }

@app.get("/api/smartcity/{camera}/detected")
def get_smartcity_camera_detected(camera: str):
    """Get latest detected vehicles for a smart city camera."""
    import base64
    stream = get_smartcity_stream(camera)
    vehicles = []
    for p in stream.detected_items[-10:]:
        vehicle_crop, plate_img, plate_text, info = p
        vehicle_b64 = None
        plate_b64 = None
        if vehicle_crop is not None and vehicle_crop.size > 0:
            _, buf = cv2.imencode('.jpg', vehicle_crop)
            vehicle_b64 = base64.b64encode(buf).decode('utf-8')
        if plate_img is not None and plate_img.size > 0:
            _, buf = cv2.imencode('.jpg', plate_img)
            plate_b64 = base64.b64encode(buf).decode('utf-8')
        vehicles.append({
            "vehicle_b64": vehicle_b64,
            "plate_b64": plate_b64,
            "plate_text": plate_text,
            "info": info,
        })
    latest = vehicles[-1] if vehicles else None
    return {
        "camera": camera,
        "vehicles": vehicles,
        "latest": latest,
        "count": len(vehicles),
    }


@app.get("/api/vehicles")
def get_vehicles(
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    license_plate: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """Test database: get vehicles from image/video/webcam processing."""
    from database_test import get_test_connection, test_get_vehicles
    return test_get_vehicles(limit, offset, license_plate, start_time, end_time)


@app.get("/api/violations")
def get_violations(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    violation_type: Optional[str] = None,
):
    """Test database: get violations from image/video/webcam processing."""
    from database_test import get_test_connection, test_get_violations
    return test_get_violations(limit, offset, violation_type)


@app.get("/api/stats")
def get_stats():
    """Test database: get stats from image/video/webcam processing."""
    from database_test import test_get_stats
    return test_get_stats()


@app.get("/api/fraud")
def get_fraud_alerts(limit: int = Query(50, le=200)):
    """Test database: get fraud alerts from image/video/webcam processing."""
    from database_test import get_test_connection
    conn = get_test_connection()
    rows = conn.execute("""
        SELECT id, license_plate, vehicle_type, color, entry_time, fraud_reason
        FROM test_vehicles WHERE fraud_alert = 1 ORDER BY entry_time DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return JSONResponse(content=[dict(row) for row in rows])


# ======================== SPEED VIOLATION API ========================

@app.get("/api/speed-violations")
def get_speed_violations_api(limit: int = Query(50, le=200), license_plate: Optional[str] = None):
    """Get speed violations from database."""
    return db_get_speed_violations(limit, 0, license_plate)

@app.get("/api/speed-violation-stats")
def get_speed_violation_stats_api():
    """Get speed violation statistics."""
    return db_get_speed_violation_stats()

@app.post("/api/speed-limit")
def set_speed_limit(speed_limit: int = Form(60)):
    """Set speed limit for violation detection on all streams."""
    # Apply to webcam stream
    if camera_stream.tracker:
        camera_stream.tracker.set_speed_limit(speed_limit)
    # Apply to parking streams
    for stream in parking_streams.values():
        if stream.tracker:
            stream.tracker.set_speed_limit(speed_limit)
    # Apply to logistics streams
    for stream in logistics_streams.values():
        if stream.tracker:
            stream.tracker.set_speed_limit(speed_limit)
    # Apply to smartcity streams
    for stream in smartcity_streams.values():
        if stream.tracker:
            stream.tracker.set_speed_limit(speed_limit)
    return {"status": "ok", "speed_limit": speed_limit}


@app.post("/api/speed-calibration")
async def set_speed_calibration(request: Request):
    """
    Set Bird's Eye View calibration for accurate speed estimation.
    
    Request Body JSON:
        src_points: List of 4 source points [[x,y], ...] (trapezoid in image space)
        dst_points: List of 4 destination points [[x,y], ...] (rectangle in BEV space)
        pixels_per_meter: Calibration factor (pixels per meter in BEV space, default: 10.0)
    """
    data = await request.json()
    
    src_points = data.get("src_points", [])
    dst_points = data.get("dst_points", [])
    pixels_per_meter = float(data.get("pixels_per_meter", 10.0))
    
    if len(src_points) != 4 or len(dst_points) != 4:
        raise HTTPException(status_code=400, detail="Must provide exactly 4 source and 4 destination points")
    
    # Convert to numpy arrays
    src_array = np.array(src_points, dtype=np.float32)
    dst_array = np.array(dst_points, dtype=np.float32)
    
    # Apply to all trackers
    trackers = [camera_stream.tracker] + \
               [s.tracker for s in parking_streams.values()] + \
               [s.tracker for s in logistics_streams.values()] + \
               [s.tracker for s in smartcity_streams.values()]
    
    for tracker in trackers:
        if tracker and hasattr(tracker, 'set_bev_config'):
            tracker.set_bev_config(src_array, dst_array, pixels_per_meter)
    
    return {
        "status": "ok",
        "message": "BEV calibration updated",
        "pixels_per_meter": pixels_per_meter
    }


@app.get("/api/speed-calibration")
def get_speed_calibration():
    """Get current BEV calibration settings."""
    tracker = camera_stream.tracker
    if tracker and hasattr(tracker, 'bev_source_points') and tracker.bev_source_points is not None:
        return {
            "src_points": tracker.bev_source_points.tolist(),
            "dst_points": tracker.bev_target_points.tolist(),
            "pixels_per_meter": tracker.pixels_per_meter
        }
    return {
        "src_points": None,
        "dst_points": None,
        "pixels_per_meter": 10.0
    }

@app.get("/api/speed-limit")
def get_speed_limit():
    """Get current speed limit."""
    limit = camera_stream.tracker.speed_limit if camera_stream.tracker else 60
    return {"speed_limit": limit}


# ======================== OCR CONSOLIDATOR API ========================

@app.get("/api/ocr-consolidator/stats")
def get_ocr_consolidator_stats():
    """Get OCR consolidator statistics (active tracks, finalized plates, etc.)."""
    return ocr_consolidator.get_stats()


# ======================== REPORTS API ========================
# Report generation removed - use database APIs for data queries


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)