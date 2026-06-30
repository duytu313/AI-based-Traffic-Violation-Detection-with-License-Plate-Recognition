"""
Tracker - Object tracking classes (SimpleTracker, ByteTrackVehicleTracker)
"""
import time
import numpy as np
import threading
import cv2
from backend.src.utils.device_utils import get_device

# Streamlit runtime context - optional, only needed for streamlit apps
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except ImportError:
    def add_script_run_ctx(thread):
        """Dummy: No-op when streamlit is not available."""
        pass


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
    Ultralytics ByteTrack wrapper. If ByteTrack is not available in the current
    environment, falls back to SimpleTracker so the app can still run.
    """
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.fallback = SimpleTracker(max_lost=30)
        self.use_fallback = False

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
            for box, track_id in zip(boxes, boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                tracked.append((int(track_id), (x1, y1, x2, y2), cls_id, conf))
            return tracked
        except Exception:
            self.use_fallback = True
            return self.fallback.update(fallback_dets)


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