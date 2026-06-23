"""
ROI (Region of Interest) Detector
Phát hiện xe nằm trong vùng polygon được vẽ thủ công.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ROIConfig:
    """Cấu hình vùng ROI dạng polygon."""
    points: List[Dict[str, int]]  # [{"x": 100, "y": 200}, ...]
    name: str = "violation_zone"

    def to_numpy(self) -> np.ndarray:
        """Chuyển điểm thành numpy array cho OpenCV."""
        return np.array([[p["x"], p["y"]] for p in self.points], dtype=np.int32)


class ROIDetector:
    """
    Phát hiện xe có nằm trong vùng ROI không.
    Sử dụng point-in-polygon test.
    """

    def __init__(self, roi_config: Optional[ROIConfig] = None):
        self.config = roi_config
        self.violation_history: Dict[int, List[bool]] = {}  # track_id -> list of in_roi flags

    def set_roi(self, points: List[Dict[str, int]], name: str = "violation_zone"):
        """Cập nhật vùng ROI."""
        self.config = ROIConfig(points=points, name=name)

    def clear_roi(self):
        """Xóa vùng ROI."""
        self.config = None

    def is_point_in_polygon(self, x: float, y: float) -> bool:
        """
        Kiểm tra điểm có nằm trong polygon không.
        Sử dụng thuật toán ray casting.
        """
        if self.config is None or len(self.config.points) < 3:
            return False

        polygon = self.config.to_numpy()
        point = (x, y)

        # Sử dụng OpenCV pointPolygonTest
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0

    def is_bbox_in_roi(self, bbox: tuple, threshold: float = 0.3) -> bool:
        """
        Kiểm tra xe (bbox) có nằm trong ROI không.
        Kiểm tra nhiều điểm trên bbox (center, corners, midpoints).
        
        Args:
            bbox: (x1, y1, x2, y2)
            threshold: tỷ lệ tối thiểu số điểm trong ROI để coi là trong vùng
            
        Returns:
            True nếu đủ điểm trong ROI
        """
        if self.config is None or len(self.config.points) < 3:
            return False

        x1, y1, x2, y2 = bbox
        
        # Kiểm tra nhiều điểm trên bbox
        test_points = [
            # Center
            ((x1 + x2) / 2, (y1 + y2) / 2),
            # Corners
            (x1, y1), (x2, y1), (x1, y2), (x2, y2),
            # Midpoints
            ((x1 + x2) / 2, y1), ((x1 + x2) / 2, y2),
            (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2),
        ]

        inside_count = sum(1 for px, py in test_points if self.is_point_in_polygon(px, py))
        return inside_count / len(test_points) >= threshold

    def process_vehicles(self, vehicles: List[Dict], red_light_active: bool = False, min_history: int = 1) -> List[Dict]:
        """
        Xử lý danh sách xe và phát hiện vi phạm ROI.
        
        Args:
            vehicles: list các dict với keys: bbox, track_id, class_name, conf
            red_light_active: True nếu đèn đỏ đang bật
            min_history: số lần liên tiếp xe phải trong ROI để coi là vi phạm
                         (1 cho ảnh đơn lẻ, 2 cho video để giảm false positive)
            
        Returns:
            list vi phạm phát hiện được
        """
        if self.config is None or len(self.config.points) < 3:
            return []

        violations = []

        for vehicle in vehicles:
            bbox = vehicle.get("bbox")
            track_id = vehicle.get("track_id", 0)
            
            if bbox is None or len(bbox) != 4:
                continue

            in_roi = self.is_bbox_in_roi(bbox)

            # Lưu lịch sử
            if track_id not in self.violation_history:
                self.violation_history[track_id] = []
            self.violation_history[track_id].append(in_roi)

            # Phát hiện vi phạm: xe vào ROI khi đèn đỏ
            if red_light_active and in_roi:
                history = self.violation_history[track_id]
                if len(history) >= min_history and sum(history[-min_history:]) >= min_history:
                    violations.append({
                        "bbox": bbox,
                        "track_id": track_id,
                        "class_name": vehicle.get("class_name", "vehicle"),
                        "conf": vehicle.get("conf", 0.0),
                        "details": "Vượt đèn đỏ (ROI)",
                        "violation_type": "RED_LIGHT_VIOLATION",
                    })

        # Dọn dẹp lịch sử cũ
        self._cleanup_history()

        return violations

    def _cleanup_history(self, max_history: int = 100):
        """Dọn dẹp lịch sử cũ."""
        for track_id in list(self.violation_history.keys()):
            if len(self.violation_history[track_id]) > max_history:
                self.violation_history[track_id] = self.violation_history[track_id][-max_history:]

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ vùng ROI lên frame."""
        if self.config is None or len(self.config.points) < 3:
            return frame

        polygon = self.config.to_numpy()

        # Vẽ polygon fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (0, 255, 255, 50))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Vẽ đường viền
        cv2.polylines(frame, [polygon], True, (0, 255, 255), 3)

        # Vẽ tên vùng
        if len(self.config.points) > 0:
            first_point = self.config.points[0]
            cv2.putText(
                frame,
                self.config.name,
                (first_point["x"], first_point["y"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        return frame

    def get_config(self) -> Optional[Dict]:
        """Lấy cấu hình ROI hiện tại."""
        if self.config is None:
            return None
        return {
            "points": self.config.points,
            "name": self.config.name,
        }

    def reset(self):
        """Reset trạng thái."""
        self.violation_history.clear()