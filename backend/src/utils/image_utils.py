"""
Image Utilities - Các hàm bổ trợ xử lý ảnh, bounding box, crop
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Tính Intersection over Union giữa 2 bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def crop_vehicle_context(image: np.ndarray, bbox: Tuple[int, int, int, int], vehicle_type: str):
    """
    Crop có ngữ cảnh. Với xe máy, bbox COCO thường chỉ ôm phần xe,
    nên cần mở rộng lên trên và hai bên để chứa người ngồi, mũ và tay.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = image.shape[:2]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    if vehicle_type == "motorcycle":
        pad_left = int(width * 0.45)
        pad_right = int(width * 0.45)
        pad_top = int(height * 1.20)
        pad_bottom = int(height * 0.25)
    else:
        pad_left = pad_right = pad_top = pad_bottom = 0

    cx1 = max(0, x1 - pad_left)
    cy1 = max(0, y1 - pad_top)
    cx2 = min(w_img, x2 + pad_right)
    cy2 = min(h_img, y2 + pad_bottom)
    crop = image[cy1:cy2, cx1:cx2]
    return crop, (cx1, cy1, cx2, cy2)


def normalize_vehicle_tuple(vehicle: tuple):
    """Chuẩn hóa vehicle tuple về 6 phần tử (bbox, vtype, color, conf, crop, crop_bbox)."""
    if len(vehicle) == 6:
        return vehicle
    if len(vehicle) == 5:
        bbox, vtype, color, conf, crop = vehicle
        return bbox, vtype, color, conf, crop, bbox
    bbox, vtype, color = vehicle[:3]
    conf = vehicle[3] if len(vehicle) > 3 else 1.0
    return bbox, vtype, color, conf, None, bbox


def build_tracking_dets(vehicles: list, vehicle_classes: dict = None):
    """Xây dựng detection list cho tracker từ vehicles list."""
    if vehicle_classes is None:
        vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    dets = []
    for vehicle in vehicles:
        bbox, vtype, _color, conf, _crop, _crop_bbox = normalize_vehicle_tuple(vehicle)
        cls_id = next((k for k, v in vehicle_classes.items() if v == vtype), 2)
        dets.append((*bbox, cls_id, conf))
    return dets


def find_matched_vehicle_by_bbox(matched: list, bbox: tuple, min_iou: float = 0.35):
    """Tìm vehicle matched gần nhất với bbox dựa trên IoU."""
    best = None
    best_iou = 0.0
    for item in matched:
        iou = bbox_iou(bbox, item[0])
        if iou > best_iou:
            best_iou = iou
            best = item
    return best if best is not None and best_iou >= min_iou else None


def find_vehicle_context_by_bbox(vehicles: list, matched: list, bbox: tuple, min_iou: float = 0.20):
    """Tìm vehicle context (bbox, vtype, color, plate_img, plate_text) từ bbox."""
    best_vehicle = None
    best_iou = 0.0
    for vehicle in vehicles:
        vehicle_bbox, vtype, color, _conf, _crop, _crop_bbox = normalize_vehicle_tuple(vehicle)
        iou = bbox_iou(bbox, vehicle_bbox)
        if iou > best_iou:
            best_iou = iou
            best_vehicle = (vehicle_bbox, vtype, color)

    if best_vehicle is None or best_iou < min_iou:
        matched_item = find_matched_vehicle_by_bbox(matched, bbox, min_iou=0.10)
        return matched_item

    vehicle_bbox, vtype, color = best_vehicle
    plate_img = None
    plate_text = ""
    for matched_item in matched:
        if matched_item[0] == vehicle_bbox:
            plate_img = matched_item[3]
            plate_text = matched_item[4]
            break
    return vehicle_bbox, vtype, color, plate_img, plate_text


def has_person_in_crop(recognizer, crop: np.ndarray, min_conf: float = 0.30) -> bool:
    """
    Kiểm tra xem có người (person) trong crop của xe máy không.
    Chỉ chạy helmet/no-helmet detection nếu có người thật sự.
    """
    try:
        results = recognizer.yolo_vehicle.predict(
            crop, device='cpu', classes=[0],  # class 0 = person in COCO
            conf=min_conf
        )[0]
        if results.boxes is not None and len(results.boxes) > 0:
            return True
    except Exception:
        pass
    return False


def build_hierarchical_violation_map(recognizer, vehicles: list, image: np.ndarray = None, debug: bool = False) -> dict:
    """
    Gom lỗi vi phạm theo từng xe máy.

    Ưu tiên thuật toán full-frame: chạy model lỗi trên toàn ảnh một lần, sau đó
    gán từng lỗi vào xe máy gần nhất theo tâm đáy. Cách này khớp với mô hình
    multi-violation và tránh mất lỗi khi crop xe máy không chứa đủ người/ngữ cảnh.

    Nếu không truyền image, fallback về cách crop từng xe để tương thích ngược.
    """
    vehicle_violation_map = {tuple(normalize_vehicle_tuple(v)[0]): [] for v in vehicles}

    if image is not None and getattr(image, "size", 0) > 0:
        all_violations = recognizer.detect_violations_on_full_image(image, debug=debug)
        return assign_violations_to_vehicle_unified(all_violations, vehicles)

    for vehicle in vehicles:
        bbox, vtype, _color, _conf, crop, crop_bbox = normalize_vehicle_tuple(vehicle)
        if vtype != "motorcycle":
            continue

        local_violations = recognizer.detect_violations_on_vehicle(crop, debug=debug)
        for vtype_v, details, (lx1, ly1, lx2, ly2), vconf in local_violations:
            gx1 = lx1 + crop_bbox[0]
            gy1 = ly1 + crop_bbox[1]
            gx2 = lx2 + crop_bbox[0]
            gy2 = ly2 + crop_bbox[1]
            vehicle_violation_map[tuple(bbox)].append((vtype_v, details, (gx1, gy1, gx2, gy2), vconf))
    return vehicle_violation_map

def merge_violation_data(vehicle_violation_map: dict, vehicles: list, matched: list) -> list:
    """Gộp dữ liệu vi phạm vào danh sách xe để hiển thị."""
    all_violation_data = []
    for vehicle in vehicles:
        bbox, vtype, color, _conf, _crop, _crop_bbox = normalize_vehicle_tuple(vehicle)
        violations = vehicle_violation_map.get(tuple(bbox), [])
        if not violations:
            continue
        plate_text = next((m[4] for m in matched if m[0] == bbox), "")
        all_violation_data.append((bbox, vtype, color, plate_text, tuple(violations)))
    return all_violation_data


def assign_violations_to_vehicle_unified(all_violations_global: list, vehicles: list) -> dict:
    """
    Gán mỗi violation vào xe máy gần nhất theo tâm đáy.
    Trả về format tuple mà merge_violation_data/main.py đang sử dụng.
    """
    normalized_vehicles = [normalize_vehicle_tuple(v) for v in vehicles]
    vehicle_violation_map = {tuple(v[0]): [] for v in normalized_vehicles}

    for viol in all_violations_global:
        best_vehicle_bbox = None
        best_score = float('inf')

        for v_bbox, v_vtype, _color, _conf, _crop, _crop_bbox in normalized_vehicles:
            if v_vtype != 'motorcycle':
                continue

            vx1, vy1, vx2, vy2 = v_bbox
            vbx1, vby1, vbx2, vby2 = viol['bbox']
            xi1, yi1 = max(vx1, vbx1), max(vy1, vby1)
            xi2, yi2 = min(vx2, vbx2), min(vy2, vby2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            viol_area = max(0, vbx2 - vbx1) * max(0, vby2 - vby1)
            veh_area = max(0, vx2 - vx1) * max(0, vy2 - vy1)
            union_area = veh_area + viol_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            veh_bottom = ((vx1 + vx2) / 2.0, float(vy2))
            viol_bottom = viol['bottom_center']
            dist = np.sqrt((viol_bottom[0] - veh_bottom[0])**2 + (viol_bottom[1] - veh_bottom[1])**2)
            score = dist * 0.5 if iou > 0.1 else dist

            if score < best_score:
                best_score = score
                best_vehicle_bbox = tuple(v_bbox)

        if best_vehicle_bbox is not None:
            vehicle_violation_map[best_vehicle_bbox].append((
                viol['type'],
                viol['details'],
                viol['bbox'],
                viol['conf'],
            ))

    return vehicle_violation_map