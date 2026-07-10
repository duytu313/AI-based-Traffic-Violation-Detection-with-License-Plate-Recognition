"""
Tracker - Object tracking classes (SimpleTracker, ByteTrackVehicleTracker)
"""
import time
import numpy as np
import threading
import cv2
from collections import defaultdict, deque
from backend.src.utils.device_utils import get_device

# Streamlit runtime context - optional, only needed for streamlit apps
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except ImportError:
    def add_script_run_ctx(thread):
        """Dummy: No-op when streamlit is not available."""
        pass


class ViewTransformer:
    """
    A class to transform points from source perspective to bird's eye view.
    
    This uses perspective transformation to convert coordinates from the camera
    view to a top-down (bird's eye) view for accurate speed calculation.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Initialize the ViewTransformer.
        
        Args:
            source: Source polygon coordinates in camera view
            target: Target polygon coordinates in bird's eye view
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from source to target perspective.
        
        Args:
            points: Array of points to transform
            
        Returns:
            Transformed points
        """
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class SimpleTracker:
    """Simple tracker based on IoU. ByteTrack-style."""
    def __init__(self, iou_thresh=0.3, max_lost=15):
        self.next_id = 1
        self.tracks = {}
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.violation_history = {}  # ID -> Boolean (Has violated before)

    def iou(self, a, b):
        x1, y1, x2, y2 = a
        x1g, y1g, x2g, y2g = b
        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def update(self, dets):
        if not dets:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['lost'] += 1
            return []
        used_det = [False] * len(dets)
        used_trk = [False] * len(self.tracks)
        trk_ids = list(self.tracks.keys())
        matched = []
        for i, tid in enumerate(trk_ids):
            best_iou = 0
            best_det = -1
            for j, det in enumerate(dets):
                if used_det[j]:
                    continue
                iou_val = self.iou(self.tracks[tid]['bbox'], det[:4])
                if iou_val > best_iou and iou_val > self.iou_thresh:
                    best_iou = iou_val
                    best_det = j
            if best_det != -1:
                matched.append((tid, best_det))
                used_trk[i] = True
                used_det[best_det] = True
                self.tracks[tid]['bbox'] = dets[best_det][:4]
                self.tracks[tid]['lost'] = 0
        for j, det in enumerate(dets):
            if not used_det[j]:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {'bbox': det[:4], 'lost': 0}
                matched.append((new_id, j))
        for i, tid in enumerate(trk_ids):
            if not used_trk[i]:
                self.tracks[tid]['lost'] += 1
        # Remove tracks lost for too long
        lost_ids = [tid for tid, info in self.tracks.items() if info['lost'] > self.max_lost]
        for tid in lost_ids:
            del self.tracks[tid]
        result = []
        for tid, det_idx in matched:
            bbox = dets[det_idx][:4]
            cls_id = dets[det_idx][4]
            conf = dets[det_idx][5]
            result.append((tid, bbox, cls_id, conf))
        return result


class ByteTrackVehicleTracker:
    """
    Ultralytics ByteTrack wrapper with Bird's Eye View speed estimation.
    If ByteTrack is not available, falls back to SimpleTracker.
    """
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.fallback = SimpleTracker(max_lost=30)
        self.use_fallback = False
        self.track_history = {}  # track_id -> list of (timestamp, bbox_center)
        self.speed_limit = 60  # km/h, default speed limit
        
        # Bird's Eye View transformation for accurate speed calculation
        # Default configuration - should be calibrated for each camera setup
        self.view_transformer = None
        self.bev_coordinates = defaultdict(lambda: deque(maxlen=30))  # track_id -> deque of (timestamp, y_bev)
        self.pixels_per_meter = 10.0  # Calibration factor: pixels per meter in BEV space
        
        # Default BEV configuration (can be updated via set_bev_config)
        self.bev_source_points = None
        self.bev_target_points = None

    def set_bev_config(self, source_points: np.ndarray, target_points: np.ndarray, pixels_per_meter: float = 10.0):
        """
        Set Bird's Eye View configuration for accurate speed calculation.
        
        Args:
            source_points: 4 points in camera view (trapezoid) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            target_points: 4 points in BEV space (rectangle) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            pixels_per_meter: Calibration factor (how many pixels represent 1 meter in BEV space)
        """
        self.bev_source_points = source_points.astype(np.float32)
        self.bev_target_points = target_points.astype(np.float32)
        self.pixels_per_meter = pixels_per_meter
        self.view_transformer = ViewTransformer(self.bev_source_points, self.bev_target_points)
        
        # Clear BEV coordinates when config changes
        self.bev_coordinates.clear()

    def _get_bottom_center(self, bbox):
        """Get bottom center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return np.array([[(x1 + x2) / 2.0, y2]])

    def update(self, frame, fallback_dets):
        if self.use_fallback:
            return self.fallback.update(fallback_dets)
        try:
            results = self.recognizer.yolo_vehicle.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                device=get_device(),
                classes=list(self.recognizer.vehicle_classes.keys()),
                conf=self.recognizer.vehicle_conf,
                verbose=False,
            )
            if not results or results[0].boxes is None or results[0].boxes.id is None:
                return self.fallback.update(fallback_dets)

            tracked = []
            boxes = results[0].boxes
            current_time = time.time()
            
            for box, track_id in zip(boxes, boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                tid = int(track_id)
                
                # Store original image coordinates for tracking
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                
                # Update track history for fallback speed calculation
                if tid not in self.track_history:
                    self.track_history[tid] = []
                
                self.track_history[tid].append((current_time, center_x, center_y))
                
                # Keep only last 30 frames (about 1.5 seconds at 20 FPS)
                if len(self.track_history[tid]) > 30:
                    self.track_history[tid].pop(0)
                
                # Update BEV coordinates if transformer is configured
                if self.view_transformer is not None:
                    bottom_center = self._get_bottom_center((x1, y1, x2, y2))
                    bev_point = self.view_transformer.transform_points(bottom_center)
                    
                    if bev_point.size > 0:
                        bev_y = int(bev_point[0][1])  # Y coordinate in BEV space
                        self.bev_coordinates[tid].append((current_time, bev_y))
                
                tracked.append((tid, (x1, y1, x2, y2), cls_id, conf))
            return tracked
        except Exception:
            self.use_fallback = True
            return self.fallback.update(fallback_dets)
    
    def calculate_speed(self, track_id):
        """
        Calculate speed for a tracked vehicle in km/h using Bird's Eye View.
        
        If BEV is configured, uses perspective-transformed coordinates for accurate speed.
        Otherwise, falls back to simple pixel-based calculation.
        
        Args:
            track_id: The track ID of the vehicle
        
        Returns:
            Speed in km/h, or None if cannot calculate
        """
        # Try BEV-based speed calculation first (more accurate)
        if self.view_transformer is not None and track_id in self.bev_coordinates:
            return self._calculate_speed_bev(track_id)
        
        # Fallback to simple pixel-based calculation
        return self._calculate_speed_simple(track_id)
    
    def _calculate_speed_bev(self, track_id):
        """
        Calculate speed using Bird's Eye View coordinates (more accurate).
        
        Uses Y-axis movement in BEV space where perspective distortion is removed.
        """
        if track_id not in self.bev_coordinates or len(self.bev_coordinates[track_id]) < 2:
            return None
        
        coords = self.bev_coordinates[track_id]
        
        # Need at least 0.5 seconds of data (10 frames at 20 FPS)
        if len(coords) < 10:
            return None
        
        # Get first and last positions
        t1, y1 = coords[0]
        t2, y2 = coords[-1]
        
        time_diff = t2 - t1
        if time_diff <= 0:
            return None
        
        # Calculate distance in BEV space (pixels)
        pixel_distance = abs(y2 - y1)
        
        # Convert to meters using calibration
        meters = pixel_distance / self.pixels_per_meter
        
        # Calculate speed in m/s, then convert to km/h
        speed_ms = meters / time_diff
        speed_kmh = speed_ms * 3.6
        
        # Sanity check: reject unrealistic speeds (max 200 km/h)
        if speed_kmh > 200:
            return None
        
        return round(speed_kmh, 1)
    
    def _calculate_speed_simple(self, track_id, pixels_per_meter=10):
        """
        Simple fallback speed calculation using pixel coordinates.
        
        Args:
            track_id: The track ID of the vehicle
            pixels_per_meter: Calibration factor (pixels per meter in the scene)
        
        Returns:
            Speed in km/h, or None if cannot calculate
        """
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return None
        
        history = self.track_history[track_id]
        
        # Use last 10 frames for more stable speed calculation
        if len(history) < 10:
            return None
        
        # Get first and last positions
        t1, x1, y1 = history[0]
        t2, x2, y2 = history[-1]
        
        time_diff = t2 - t1
        if time_diff <= 0:
            return None
        
        # Calculate pixel distance
        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Convert to meters
        meters = pixel_distance / pixels_per_meter
        
        # Calculate speed in m/s, then convert to km/h
        speed_ms = meters / time_diff
        speed_kmh = speed_ms * 3.6
        
        # Sanity check: reject unrealistic speeds (max 200 km/h)
        if speed_kmh > 200:
            return None
        
        return round(speed_kmh, 1)
    
    def set_speed_limit(self, speed_limit):
        """Set speed limit for violation detection."""
        self.speed_limit = speed_limit
    
    def get_speed(self, track_id):
        """Get current speed of a tracked vehicle."""
        return self.calculate_speed(track_id)


class VideoCaptureThread:
    """Separate video/camera reading thread to avoid blocking Streamlit."""
    def __init__(self, src=0, backend=cv2.CAP_ANY):
        self.src = src
        self.backend = backend
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.error = None

    def start(self):
        try:
            self.cap = cv2.VideoCapture(self.src, self.backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open {self.src}")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.running = True
            thread = threading.Thread(target=self._run, daemon=True)
            add_script_run_ctx(thread)
            thread.start()
        except Exception as e:
            self.error = str(e)
            raise

    def _run(self):
        while self.running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame = frame
        if self.cap:
            self.cap.release()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()