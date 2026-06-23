"""
Engine - Lớp LicensePlateRecognizer (Logic nhận diện & Vi phạm)
Phụ trách: detect_vehicles, detect_plates, match_plates_to_vehicles,
classify_color, detect_violations, traffic light detection, drawing.
"""
import cv2
import numpy as np
import re
import os
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer as FastPlateOCR
from typing import Tuple, List, Dict, Optional

# Zone-based red light detection
from backend.src.core.zones import RedLightZoneDetector, ZoneConfig

# Streamlit is optional - only needed for legacy streamlit app
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    # Create dummy st for compatibility when streamlit is not installed
    class _DummySt:
        def cache_resource(self, func):
            return func
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def sidebar(self): return self
        def success(self, msg): print(f"SUCCESS: {msg}")
        def write(self, msg): print(msg)
        def info(self, msg): print(f"INFO: {msg}")
        def spinner(self, msg):
            class _Spinner:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return _Spinner()
        def markdown(self, msg): print(msg)
        def text_input(self, label, *args, **kwargs): return ""
        def selectbox(self, label, *args, **kwargs): return ""
        def checkbox(self, label, *args, **kwargs): return False
        def slider(self, label, *args, **kwargs): return 1
        def radio(self, label, *args, **kwargs): return ""
        def button(self, label): return False
        def columns(self, n): return [self] * n
        def empty(self): return self
        def image(self, *args, **kwargs): pass
        def header(self, msg): pass
        def subheader(self, msg): pass
        def divider(self): pass
        def caption(self, msg): pass
        def set_page_config(self, **kwargs): pass
        def title(self, msg): pass
        def file_uploader(self, *args, **kwargs): return None
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    st = _DummySt()


def load_models(
    yolo_plate_path="files_model/license_plate_detector.pt",
    yolo_vehicle_path="yolov8n.pt",
    color_model_path="files_model/vehicle_color_n_cls.pt",
    helmet_model_path="files_model/helmet.pt",
    traffic_light_model_path="files_model/light_traffic.pt"
):
    """Tải tất cả các model YOLO và OCR engine."""
    yolo_plate = YOLO(yolo_plate_path)
    yolo_vehicle = YOLO(yolo_vehicle_path)
    try:
        color_model = YOLO(color_model_path)
        if not hasattr(color_model, 'predict'):
            raise ValueError
    except:
        st.error("Không thể load model màu. Chức năng màu sẽ bị vô hiệu.")
        color_model = None

    try:
        helmet_model = YOLO(helmet_model_path)
        if not hasattr(helmet_model, 'predict'):
            raise ValueError
    except:
        st.warning("Không thể load model phát hiện vi phạm (helmet.pt). Chức năng phát hiện vi phạm sẽ bị vô hiệu.")
        helmet_model = None

    try:
        traffic_light_model = YOLO(traffic_light_model_path)
        if not hasattr(traffic_light_model, 'predict'):
            raise ValueError
    except:
        st.warning("Không thể load model đèn giao thông (light_traffic.pt). Chức năng vượt đèn đỏ sẽ bị vô hiệu.")
        traffic_light_model = None

    ocr_engine = FastPlateOCR(hub_ocr_model='cct-s-v2-global-model', device='auto')
    return yolo_plate, yolo_vehicle, color_model, helmet_model, traffic_light_model, ocr_engine


class LicensePlateRecognizer:
    """Lớp nhận diện chính: phát hiện xe, biển số, màu sắc, vi phạm, đèn giao thông."""

    def __init__(self, yolo_plate, yolo_vehicle, color_model, helmet_model,
                 traffic_light_model, ocr_engine, vehicle_conf=0.25, plate_conf=0.1):
        self.yolo_plate = yolo_plate
        self.yolo_vehicle = yolo_vehicle
        self.color_model = color_model
        self.helmet_model = helmet_model
        self.traffic_light_model = traffic_light_model
        self.ocr_engine = ocr_engine
        self.vehicle_conf = vehicle_conf
        self.plate_conf = plate_conf
        self.color_names = ["beige", "black", "blue", "brown", "gold", "green",
                            "grey", "orange", "pink", "purple", "red", "silver",
                            "tan", "white", "yellow"]
        self.vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.traffic_road_user_classes = {"biker", "car", "pedestrian", "truck"}
        self.traffic_light_conf = 0.25
        # Ngưỡng cho phát hiện vi phạm
        self.violation_conf_more_than_two = 0.50
        self.violation_conf_using_mobile = 0.25
        self.violation_conf_without_helmet = 0.25
        self.track_violation_status = {}

        # Zone-based red light detection
        default_config = ZoneConfig()
        self.zone_detector = RedLightZoneDetector(default_config)

    # ======================== SETTERS ========================

    def set_vehicle_conf(self, conf):
        self.vehicle_conf = conf

    def set_plate_conf(self, conf):
        self.plate_conf = conf

    def set_violation_conf_limit(self, conf):
        self.violation_conf_more_than_two = conf
        self.violation_conf_using_mobile = conf
        self.violation_conf_without_helmet = conf

    def set_traffic_light_conf(self, conf):
        self.traffic_light_conf = conf

    # ======================== VEHICLE DETECTION ========================

    def detect_vehicles(self, image, debug=False):
        """Phát hiện phương tiện trong ảnh."""
        results = self.yolo_vehicle.predict(
            image, device='cpu', classes=list(self.vehicle_classes.keys()),
            conf=self.vehicle_conf
        )[0]
        vehicles = []
        if debug:
            st.sidebar.write(f"🔍 Tổng số xe phát hiện: {len(results.boxes) if results.boxes else 0}")
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                vehicle_type = self.vehicle_classes.get(cls_id, "vehicle")

                from src.utils.image_utils import crop_vehicle_context
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

                if debug:
                    st.sidebar.write(f"   - {vehicle_type} (conf={conf:.2f})")
        return vehicles

    # ======================== PLATE DETECTION & OCR ========================

    def detect_plates(self, image):
        """Phát hiện biển số trong ảnh."""
        results = self.yolo_plate.predict(image, device='cpu', conf=self.plate_conf)[0]
        plates = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            plate_img = image[y1:y2, x1:x2]
            plates.append((plate_img, (x1, y1, x2, y2)))
        return plates

    def extract_text(self, plate_img):
        """Trích xuất văn bản từ ảnh biển số bằng OCR."""
        if plate_img is None or plate_img.size == 0:
            return ""
        image_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        result = self.ocr_engine.run(image_rgb)
        raw_text = ""
        if hasattr(result, 'text'):
            raw_text = result.text
        elif isinstance(result, str):
            raw_text = result
        elif isinstance(result, list) and len(result) > 0:
            first = result[0]
            if hasattr(first, 'text'):
                raw_text = first.text
            else:
                raw_text = str(first)
        else:
            raw_text = str(result) if result else ""

        keywords = ['PREDICTION', 'PLATE', 'CHARPROBS', 'CHARS', 'REGION',
                    'UNITEDKINGDOM', 'VIETNAM', 'NONE', 'PROB', 'DETECTION', 'CONFIDENCE']
        for kw in keywords:
            raw_text = re.sub(kw, '', raw_text, flags=re.IGNORECASE)

        cleaned = re.sub(r'[^A-Z0-9\-.]', '', raw_text.upper())
        pattern_vn = r'(\d{1,2}[A-Z]{1,2}[-\d\.]*\d+)'
        match = re.search(pattern_vn, cleaned)
        if match:
            return match.group(1)
        pattern_eu = r'([A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3})'
        match = re.search(pattern_eu, cleaned)
        if match:
            return match.group(1)
        match_fb = re.search(r'([A-Z0-9]{4,})', cleaned)
        if match_fb:
            return match_fb.group(1)
        return cleaned if cleaned else ""

    # ======================== COLOR CLASSIFICATION ========================

    def classify_color(self, vehicle_img):
        """Phân loại màu sắc của xe."""
        if vehicle_img is None or vehicle_img.size == 0 or self.color_model is None:
            return "unknown"
        results = self.color_model.predict(vehicle_img, device='cpu', verbose=False)
        if len(results) == 0:
            return "unknown"
        if hasattr(results[0], 'probs') and results[0].probs is not None:
            top1 = results[0].probs.top1
            if top1 < len(self.color_names):
                return self.color_names[top1]
        elif hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes.cls[0])
            if cls_id < len(self.color_names):
                return self.color_names[cls_id]
        return "unknown"

    # ======================== PLATE-VEHICLE MATCHING ========================

    def match_plates_to_vehicles(self, vehicles, plates):
        """Ghép biển số với xe dựa trên tọa độ."""
        matched = []
        for v in vehicles:
            if len(v) == 6:
                bbox, vtype, color, conf, vehicle_crop, crop_bbox = v
            elif len(v) == 5:
                bbox, vtype, color, conf, vehicle_crop = v
            elif len(v) == 4:
                bbox, vtype, color, conf = v
            else:
                bbox, vtype, color = v
                conf = 1.0
            vx1, vy1, vx2, vy2 = bbox
            for plate_img, (px1, py1, px2, py2) in plates:
                if (px1 >= vx1 and px2 <= vx2 and py1 >= vy1 and py2 <= vy2):
                    plate_text = self.extract_text(plate_img)
                    if plate_text:
                        matched.append((bbox, vtype, color, plate_img, plate_text))
                    break
        return matched

    # ======================== VIOLATION DETECTION ========================

    def detect_violations_on_full_image(self, full_image, debug=False):
        """
        Phát hiện vi phạm trên TOÀN BỘ ẢNH (full image) - MỘT LẦN DUY NHẤT.
        Trả về: list các dict với keys: type, details, bbox (global), conf, bottom_center
        """
        violations = []
        if self.helmet_model is None or full_image is None or full_image.size == 0:
            return violations

        results = self.helmet_model.predict(
            full_image, device='cpu', conf=0.15, iou=0.45,
            agnostic_nms=True, verbose=False
        )

        if len(results) == 0 or results[0].boxes is None:
            return violations

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            conf_val = float(box.conf[0])
            cn = class_name.strip().lower()

            is_valid = False
            viol_type = None
            details = None

            if 'more_than_two_persons' in cn or 'more_than' in cn:
                if conf_val >= self.violation_conf_more_than_two:
                    is_valid = True
                    viol_type = 'MORE_THAN_TWO_PERSONS'
                    details = 'Xe chở quá 2 người'
            elif 'without_helmet' in cn or 'no_helmet' in cn or 'w/o_helmet' in cn:
                if conf_val >= self.violation_conf_without_helmet:
                    is_valid = True
                    viol_type = 'WITHOUT_HELMET'
                    details = 'Người không đội mũ bảo hiểm'
            elif 'using_mobile' in cn or 'phone' in cn or 'mobile' in cn:
                if conf_val >= self.violation_conf_using_mobile:
                    is_valid = True
                    viol_type = 'USING_MOBILE'
                    details = 'Sử dụng điện thoại khi lái xe'

            if is_valid:
                bottom_cx = (x1 + x2) / 2.0
                bottom_cy = float(y2)
                violations.append({
                    'type': viol_type,
                    'details': details,
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf_val,
                    'bottom_center': (bottom_cx, bottom_cy),
                })

        return violations

    def detect_violations_on_vehicle(self, vehicle_img, debug=False):
        """
        Phát hiện vi phạm trên ảnh xe đã crop (legacy - giữ lại để tương thích).
        Trả về: list các tuple (violation_type, details, bbox, conf)
        """
        violations = []
        if self.helmet_model is None or vehicle_img is None or vehicle_img.size == 0:
            return violations

        raw_conf = min(0.05, self.violation_conf_more_than_two,
                       self.violation_conf_without_helmet,
                       self.violation_conf_using_mobile)
        results = self.helmet_model.predict(
            vehicle_img, device='cpu', conf=raw_conf, iou=0.50,
            agnostic_nms=True, verbose=False
        )

        if len(results) == 0 or results[0].boxes is None:
            return violations

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            conf_val = float(box.conf[0])
            cn = class_name.strip().lower()

            if debug:
                st.sidebar.write(f"      violation candidate: {class_name} ({conf_val:.2f})")

            if 'more_than_two_persons' in cn or 'more_than' in cn:
                if conf_val >= self.violation_conf_more_than_two:
                    violations.append(('MORE_THAN_TWO_PERSONS', 'Xe chở quá 2 người',
                                       (x1, y1, x2, y2), conf_val))
            elif 'without_helmet' in cn or 'no_helmet' in cn or 'w/o_helmet' in cn:
                if conf_val >= self.violation_conf_without_helmet:
                    violations.append(('WITHOUT_HELMET', 'Người không đội mũ bảo hiểm',
                                       (x1, y1, x2, y2), conf_val))
            elif 'using_mobile' in cn or 'phone' in cn or 'mobile' in cn:
                if conf_val >= self.violation_conf_using_mobile:
                    violations.append(('USING_MOBILE', 'Sử dụng điện thoại khi lái xe',
                                       (x1, y1, x2, y2), conf_val))

        return violations

    def assign_violations_to_vehicle(self, violations_global, vehicle_bbox, search_radius=100, debug=False):
        """Gán các vi phạm vào xe dựa trên khoảng cách và IoU."""
        assigned = []
        veh_bottom_cx = (vehicle_bbox[0] + vehicle_bbox[2]) / 2.0
        veh_bottom_cy = float(vehicle_bbox[3])
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vehicle_area = (vx2 - vx1) * (vy2 - vy1)

        for viol in violations_global:
            vbx1, vby1, vbx2, vby2 = viol['bbox']

            # Tính IoU
            xi1, yi1 = max(vx1, vbx1), max(vy1, vby1)
            xi2, yi2 = min(vx2, vbx2), min(vy2, vby2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            viol_area = (vbx2 - vbx1) * (vby2 - vby1)
            union_area = vehicle_area + viol_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            # Tính khoảng cách
            viol_bottom_cx = (vbx1 + vbx2) / 2.0
            viol_bottom_cy = float(vby2)
            dist = np.sqrt((viol_bottom_cx - veh_bottom_cx)**2 +
                           (viol_bottom_cy - veh_bottom_cy)**2)

            if iou <= 0.1 and dist > search_radius:
                continue

            assigned.append({
                'type': viol['type'],
                'details': viol['details'],
                'bbox': viol['bbox'],
                'conf': viol['conf'],
                'distance': dist,
                'iou': iou,
            })

        return assigned

    @staticmethod
    def get_violation_color(vtype):
        """Trả về màu BGR tương ứng với loại vi phạm."""
        colors = {
            'MORE_THAN_TWO_PERSONS': (0, 0, 255),       # Đỏ
            'WITHOUT_HELMET': (0, 165, 255),            # Cam
            'USING_MOBILE': (255, 0, 255),              # Tím (Magenta)
            'RED_LIGHT_VIOLATION': (0, 0, 255)          # Đỏ đậm
        }
        return colors.get(vtype, (0, 0, 255))

    @staticmethod
    def get_violation_icon(vtype):
        """Trả về icon tương ứng với loại vi phạm."""
        if 'USING_MOBILE' in vtype:
            return "\U0001F4F1"  # 📱
        elif 'MORE_THAN_TWO_PERSONS' in vtype:
            return "\U0001F6F5"  # 🛵
        elif 'RED_LIGHT_VIOLATION' in vtype:
            return "\U0001F6A8"  # 🚨
        else:
            return "\u26D1\uFE0F"  # ⛑️

    def draw_violations_on_frame(self, frame, violations, offset_x=0, offset_y=0):
        """Vẽ các khung vi phạm lên frame với màu sắc tương ứng."""
        for vtype, details, (x1, y1, x2, y2), conf in violations:
            color = self.get_violation_color(vtype)
            cv2.rectangle(frame, (x1 + offset_x, y1 + offset_y),
                          (x2 + offset_x, y2 + offset_y), color, 2)
            label = f"{vtype} {conf*100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_y1 = max(y1 + offset_y, text_h + 5)
            cv2.rectangle(frame, (x1 + offset_x, text_y1 - text_h - 4),
                          (x1 + offset_x + text_w + 6, text_y1 + baseline - 2), color, -1)
            cv2.putText(frame, label, (x1 + offset_x + 3, text_y1 - 2),
                        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        return frame

    # ======================== TRAFFIC LIGHT DETECTION ========================

    def detect_traffic_scene(self, image):
        """
        Phát hiện đèn giao thông và người/phương tiện từ light_traffic.pt.
        Trả về: lights, road_users
        """
        lights = []
        road_users = []
        if self.traffic_light_model is None or image is None or image.size == 0:
            return lights, road_users

        results = self.traffic_light_model.predict(
            image, device='cpu', conf=self.traffic_light_conf, verbose=False
        )
        if len(results) == 0 or results[0].boxes is None:
            return lights, road_users

        names = results[0].names
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            normalized = cls_name.lower()

            if normalized.startswith("trafficlight"):
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                light_img = image[y1:y2, x1:x2]
                state = self.infer_traffic_light_state(cls_name, light_img)
                lights.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_name": cls_name,
                    "state": state,
                    "conf": conf
                })
            elif normalized in self.traffic_road_user_classes:
                road_users.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_name": cls_name,
                    "conf": conf
                })
        return lights, road_users

    def red_light_is_active(self, lights):
        return any(light["state"] == "red" for light in lights)

    def detect_red_light_violations(self, lights, road_users, stop_line_y, crossing_direction="down"):
        """
        Legacy method: phát hiện vượt đèn đỏ dựa trên một đường (line-based).
        Đã thay thế bằng detect_red_light_violations_zone_based().
        Giữ lại để tương thích ngược.
        """
        if not self.red_light_is_active(lights):
            return []
        violations = []
        for obj in road_users:
            x1, y1, x2, y2 = obj["bbox"]
            foot_y = y2
            head_y = y1
            crossed = foot_y >= stop_line_y if crossing_direction == "down" else head_y <= stop_line_y
            if crossed:
                violations.append({
                    "bbox": obj["bbox"],
                    "class_name": obj["class_name"],
                    "conf": obj["conf"],
                    "details": "Vượt đèn đỏ"
                })
        return violations

    def detect_red_light_violations_zone_based(self, lights, road_users, frame=None):
        """
        Phát hiện vượt đèn đỏ dựa trên VÙNG (zone-based).
        Sử dụng RedLightZoneDetector để theo dõi vị trí xe qua các frame
        và phát hiện vi phạm khi xe đi từ Waiting → Stop → Intersection.

        Args:
            lights: list đèn từ detect_traffic_scene()
            road_users: list road_users từ detect_traffic_scene()
            frame: ảnh frame hiện tại (optional, để vẽ zones)

        Returns:
            list vi phạm: [{bbox, class_name, conf, details, track_id, zone_history, "zone_based": True}]
        """
        self.zone_detector.update_from_traffic_lights(lights)
        violations = self.zone_detector.process_vehicles(road_users)

        # Gắn cờ zone_based để phân biệt với legacy
        for v in violations:
            v["zone_based"] = True

        return violations

    def set_zone_config(self, **kwargs):
        """
        Cập nhật cấu hình vùng cho zone detector.
        Ví dụ: recognizer.set_zone_config(waiting_end=150, stop_end=300, intersection_end=500)
        """
        self.zone_detector.update_config(**kwargs)

    def get_zone_stats(self):
        """Lấy thống kê zone detector hiện tại."""
        return self.zone_detector.get_stats()

    def draw_traffic_zones(self, frame):
        """Vẽ các vùng lên frame (Waiting Zone, Stop Zone, Intersection)."""
        return self.zone_detector.draw_zones(frame)

    def reset_zone_detector(self):
        """Reset zone detector về trạng thái ban đầu."""
        self.zone_detector.reset()

    def infer_traffic_light_state(self, cls_name, light_img):
        """
        Suy luận màu đèn từ class của model, fallback bằng phân tích HSV theo vùng (zone-based).
        """
        normalized = cls_name.lower().replace("_", "-").strip()
        if "red" in normalized:
            return "red"

        # Phân tích màu theo vùng (zone-based)
        zone_state = self._classify_traffic_light_by_zones(light_img)
        if zone_state != "unknown":
            return zone_state

        # Fallback: phân tích toàn bộ ảnh
        color_state, color_scores = self._classify_traffic_light_full(light_img, return_scores=True)
        red_score = color_scores.get("red", 0)
        green_score = color_scores.get("green", 0)
        yellow_score = color_scores.get("yellow", 0)
        strong_red = red_score >= 8 and red_score >= max(green_score, yellow_score) * 0.35
        if strong_red:
            return "red"

        if "yellow" in normalized:
            return "yellow"
        if "green" in normalized:
            return "green"
        return color_state

    def _classify_traffic_light_by_zones(self, light_img):
        """Chia ảnh cột đèn thành 3 phần dọc để phân tích màu HSV."""
        if light_img is None or light_img.size == 0:
            return "unknown"
        h, w = light_img.shape[:2]
        if h < 10:
            return "unknown"

        top_end = int(h * 0.30)
        mid_end = int(h * 0.65)

        top_zone = light_img[0:top_end, :]
        mid_zone = light_img[top_end:mid_end, :]
        bot_zone = light_img[mid_end:h, :]

        hsv_top = cv2.cvtColor(top_zone, cv2.COLOR_BGR2HSV) if top_zone.size > 0 else None
        hsv_mid = cv2.cvtColor(mid_zone, cv2.COLOR_BGR2HSV) if mid_zone.size > 0 else None
        hsv_bot = cv2.cvtColor(bot_zone, cv2.COLOR_BGR2HSV) if bot_zone.size > 0 else None

        bright_saturated = lambda hsv_img: (hsv_img[:, :, 1] >= 60) & (hsv_img[:, :, 2] >= 80) if hsv_img is not None else None

        def count_color(hsv_img, color_range_fn):
            if hsv_img is None:
                return 0
            bs = bright_saturated(hsv_img)
            if bs is None:
                return 0
            mask = color_range_fn(hsv_img[:, :, 0]) & bs
            return int(np.count_nonzero(mask))

        def red_range(h):
            return (h <= 10) | (h >= 170)

        def yellow_range(h):
            return (h >= 15) & (h <= 38)

        def green_range(h):
            return (h >= 40) & (h <= 95)

        top_red = count_color(hsv_top, red_range)
        mid_red = count_color(hsv_mid, red_range)
        bot_red = count_color(hsv_bot, red_range)
        top_yellow = count_color(hsv_top, yellow_range)
        mid_yellow = count_color(hsv_mid, yellow_range)
        bot_yellow = count_color(hsv_bot, yellow_range)
        top_green = count_color(hsv_top, green_range)
        mid_green = count_color(hsv_mid, green_range)
        bot_green = count_color(hsv_bot, green_range)

        top_area = top_zone.shape[0] * top_zone.shape[1]
        mid_area = mid_zone.shape[0] * mid_zone.shape[1]
        bot_area = bot_zone.shape[0] * bot_zone.shape[1]
        min_px_top = max(4, int(top_area * 0.002))
        min_px_mid = max(4, int(mid_area * 0.002))
        min_px_bot = max(4, int(bot_area * 0.002))

        if top_red >= min_px_top and top_red >= max(top_green, top_yellow) * 0.35:
            return "red"
        if mid_yellow >= min_px_mid and mid_yellow >= max(mid_red, mid_green) * 0.5:
            return "yellow"
        if bot_green >= min_px_bot and bot_green >= max(bot_red, bot_yellow) * 0.45:
            return "green"

        total_red = top_red + mid_red + bot_red
        total_green = top_green + mid_green + bot_green
        if top_red >= min_px_top and total_red >= total_green * 0.5:
            return "red"

        return "unknown"

    def _classify_traffic_light_full(self, light_img, return_scores=False):
        """Phân tích màu trên toàn bộ ảnh (fallback)."""
        empty_scores = {"red": 0, "yellow": 0, "green": 0}
        if light_img is None or light_img.size == 0:
            return ("unknown", empty_scores) if return_scores else "unknown"

        hsv = cv2.cvtColor(light_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        bright_saturated = (s >= 60) & (v >= 80)

        red_mask = (((h <= 10) | (h >= 170)) & bright_saturated)
        yellow_mask = ((h >= 15) & (h <= 38) & bright_saturated)
        green_mask = ((h >= 40) & (h <= 95) & bright_saturated)

        scores = {
            "red": int(np.count_nonzero(red_mask)),
            "yellow": int(np.count_nonzero(yellow_mask)),
            "green": int(np.count_nonzero(green_mask))
        }
        best_state, best_score = max(scores.items(), key=lambda item: item[1])
        signal_pixels = sum(scores.values())

        if signal_pixels == 0:
            return ("unknown", scores) if return_scores else "unknown"

        crop_area = light_img.shape[0] * light_img.shape[1]
        enough_pixels = best_score >= max(4, int(crop_area * 0.002))
        dominant_enough = best_score / max(1, signal_pixels) >= 0.45
        state = best_state if enough_pixels and dominant_enough else "unknown"
        return (state, scores) if return_scores else state

    def classify_traffic_light_color(self, light_img, return_scores=False):
        """Wrapper giữ tên cũ cho tương thích ngược."""
        return self._classify_traffic_light_full(light_img, return_scores=return_scores)

    def draw_traffic_scene(self, frame, lights, road_users, red_light_violations, stop_line_y=None, show_zones=False):
        """
        Vẽ đèn giao thông và vi phạm lên frame.
        
        Args:
            show_zones: Nếu True, vẽ các zone (Waiting/Stop/Intersection) lên frame
        """
        # Vẽ zones nếu được yêu cầu
        if show_zones:
            frame = self.draw_traffic_zones(frame)

        state_colors = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 200, 0),
            "unknown": (180, 180, 180)
        }
        if stop_line_y is not None and not show_zones:
            cv2.line(frame, (0, stop_line_y), (frame.shape[1], stop_line_y), (0, 0, 255), 2)
            cv2.putText(frame, "STOP LINE", (10, max(25, stop_line_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        for light in lights:
            x1, y1, x2, y2 = light["bbox"]
            color = state_colors.get(light["state"], (180, 180, 180))
            label = f"{light['state'].upper()} {light['conf']*100:.1f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for obj in red_light_violations:
            x1, y1, x2, y2 = obj["bbox"]
            color = (0, 0, 255)
            label = f"{obj['class_name']} {obj['conf']*100:.1f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame