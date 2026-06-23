"""
Red Light Zone Detector
Phát hiện vượt đèn đỏ dựa trên vùng (zone-based) thay vì đường (line-based).

Sơ đồ các vùng:

      Camera
┌─────────────────┐
│  Waiting Zone   │  - Khu vực chờ đèn (trước vạch dừng)
├─────────────────┤
│   Stop Zone     │  - Khu vực vạch dừng (critical zone)
├─────────────────┤
│ Intersection    │  - Khu vực giao lộ (đã qua vạch dừng)
└─────────────────┘

Logic phát hiện vi phạm:
  - Khi đèn đỏ:
    * Xe đi từ Waiting → Stop → Intersection => VI PHẠM
    * Xe đứng yên trong Waiting => OK (chờ đúng)
    * Xe đã ở trong Intersection khi đèn chuyển đỏ => OK (đang thoát giao lộ)
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ZoneConfig:
    """
    Cấu hình các vùng theo tọa độ Y (dọc theo chiều di chuyển của xe).
    
    direction: hướng camera
      - "down": camera nhìn từ trên xuống (top-down), xe đi từ Y nhỏ -> Y lớn
      - "up": camera nhìn từ đằng sau (rear-view), xe đi từ Y lớn -> Y nhỏ
    """
    direction: str = "down"       # "down" = top-down, "up" = rear-view
    waiting_start: int = 0        # Y bắt đầu vùng chờ
    waiting_end: int = 200        # Y kết thúc vùng chờ
    stop_start: int = 200         # Y bắt đầu vùng vạch dừng
    stop_end: int = 300           # Y kết thúc vùng vạch dừng
    intersection_start: int = 300 # Y bắt đầu vùng giao lộ
    intersection_end: int = 500   # Y kết thúc vùng giao lộ

    @property
    def stop_line_y(self) -> int:
        """Vị trí vạch dừng (trung tâm của stop zone)."""
        return (self.stop_start + self.stop_end) // 2

    def get_zone_name(self, y_bottom: int, y_top: int = None) -> str:
        """
        Xác định tên vùng dựa trên tọa độ Y và hướng camera.
        
        Với direction="down" (top-down): xe đi từ trên xuống
          - waiting: y_bottom <= waiting_end
          - stop: waiting_end < y_bottom <= stop_end
          - intersection: y_bottom > intersection_start
          
        Với direction="up" (rear-view): xe đi từ dưới lên
          - waiting: y_bottom >= waiting_end (gần camera)
          - stop: waiting_end > y_bottom >= stop_start
          - intersection: y_bottom < intersection_start (xa camera)
        """
        if self.direction == "up":
            # Đảo ngược: camera nhìn từ đằng sau
            if y_bottom >= self.waiting_end:
                return "waiting"
            elif y_bottom >= self.stop_start:
                return "stop"
            elif y_bottom <= self.intersection_start:
                return "intersection"
        else:
            # Mặc định: camera nhìn từ trên xuống
            if y_bottom <= self.waiting_end:
                return "waiting"
            elif y_bottom <= self.stop_end:
                return "stop"
            elif y_bottom >= self.intersection_start:
                return "intersection"
        return "unknown"

    def is_in_intersection(self, y_bottom: int) -> bool:
        return y_bottom >= self.intersection_start

    def is_in_stop_zone(self, y_bottom: int) -> bool:
        return self.stop_start <= y_bottom <= self.stop_end

    def is_in_waiting_zone(self, y_bottom: int) -> bool:
        return y_bottom <= self.waiting_end


class VehicleZoneState:
    """
    Theo dõi trạng thái vùng của một xe qua các frame.
    """
    def __init__(self, track_id: int, initial_zone: str):
        self.track_id = track_id
        self.zones_visited = [initial_zone]  # Lịch sử các vùng đã đi qua
        self.current_zone = initial_zone
        self.first_seen_time = cv2.getTickCount()
        self.last_seen_time = self.first_seen_time
        self.violation_detected = False
        self.violation_frame = None
        self.positions_history: List[Tuple[int, int]] = []  # (y_bottom, frame_count)

    def update(self, zone: str, y_bottom: int, frame_count: int = 0):
        """Cập nhật vị trí mới của xe."""
        self.last_seen_time = cv2.getTickCount()
        self.positions_history.append((y_bottom, frame_count))

        if zone != self.current_zone:
            if zone not in self.zones_visited:
                self.zones_visited.append(zone)
            self.current_zone = zone

    def has_violated_red_light(self) -> bool:
        """
        Kiểm tra vi phạm dựa trên lịch sử vùng.
        Vi phạm khi: waiting → stop → intersection (đi qua cả 3 vùng).
        """
        if self.violation_detected:
            return True

        # Kiểm tra thứ tự: phải có waiting, stop, intersection
        has_waiting = "waiting" in self.zones_visited
        has_stop = "stop" in self.zones_visited
        has_intersection = "intersection" in self.zones_visited

        if has_waiting and has_stop and has_intersection:
            # Kiểm tra thứ tự xuất hiện (không tính "unknown")
            ordered_zones = [z for z in self.zones_visited if z != "unknown"]
            if len(ordered_zones) >= 3:
                # Tìm index của từng vùng
                try:
                    idx_waiting = ordered_zones.index("waiting")
                    idx_stop = ordered_zones.index("stop")
                    idx_intersection = ordered_zones.index("intersection")
                    # Phải theo thứ tự: waiting -> stop -> intersection
                    if idx_waiting < idx_stop < idx_intersection:
                        self.violation_detected = True
                        return True
                except ValueError:
                    pass

        # Trường hợp: xe xuất hiện ở waiting rồi vào thẳng intersection
        # (vượt đèn đỏ với tốc độ cao)
        if has_waiting and has_intersection and not has_stop:
            # Check if it went from waiting directly to intersection (speeding through)
            ordered_zones = [z for z in self.zones_visited if z != "unknown"]
            if len(ordered_zones) >= 2:
                try:
                    idx_waiting = ordered_zones.index("waiting")
                    idx_intersection = ordered_zones.index("intersection")
                    if idx_waiting < idx_intersection:
                        # Kiểm tra nếu xe đi rất nhanh qua khu vực stop
                        self.violation_detected = True
                        return True
                except ValueError:
                    pass

        return False

    def is_clearing_intersection(self) -> bool:
        """
        Xe đang thoát giao lộ (đã ở intersection từ đầu).
        Không tính là vi phạm.
        """
        if len(self.zones_visited) == 1 and self.zones_visited[0] == "intersection":
            return True
        return False

    def should_be_tracked(self, max_inactive_ticks: int = 5000000) -> bool:
        """Kiểm tra xe còn đang được theo dõi hay không."""
        elapsed = cv2.getTickCount() - self.last_seen_time
        return elapsed < max_inactive_ticks


class RedLightZoneDetector:
    """
    Phát hiện vượt đèn đỏ dựa trên vùng (zone-based).
    Sử dụng ZoneConfig và VehicleZoneState để theo dõi và phát hiện vi phạm.
    """

    def __init__(self, zone_config: Optional[ZoneConfig] = None):
        self.config = zone_config or ZoneConfig()
        self.tracked_vehicles: Dict[int, VehicleZoneState] = {}
        self.frame_count = 0
        self.red_light_active = False
        self.red_light_start_frame = -1
        self.debug = False

    def update_config(self, waiting_end: int = None, stop_start: int = None,
                      stop_end: int = None, intersection_start: int = None,
                      intersection_end: int = None):
        """Cập nhật cấu hình vùng."""
        if waiting_end is not None:
            self.config.waiting_end = waiting_end
            self.config.stop_start = waiting_end
        if stop_end is not None:
            self.config.stop_end = stop_end
            self.config.intersection_start = stop_end
        if intersection_end is not None:
            self.config.intersection_end = intersection_end
        if stop_start is not None:
            self.config.stop_start = stop_start
        if intersection_start is not None:
            self.config.intersection_start = intersection_start

    def set_red_light(self, is_red: bool):
        """Cập nhật trạng thái đèn đỏ."""
        if is_red and not self.red_light_active:
            self.red_light_start_frame = self.frame_count
        self.red_light_active = is_red

    def update_from_traffic_lights(self, lights: List[Dict]):
        """Cập nhật trạng thái đèn từ danh sách đèn giao thông."""
        is_red = any(light.get("state") == "red" for light in lights)
        self.set_red_light(is_red)

    def process_vehicles(self, road_users_or_tracked: List[Any]) -> List[Dict]:
        """
        Xử lý danh sách xe từ traffic light detection hoặc tracker.
        Trả về danh sách vi phạm phát hiện được.

        Args:
            road_users_or_tracked: list các dict với keys: bbox, class_name, conf
                                   hoặc list các track có bbox

        Returns:
            list các dict: {bbox, class_name, conf, details, zone_history}
        """
        self.frame_count += 1
        violations = []

        current_track_ids = set()

        for obj in road_users_or_tracked:
            if isinstance(obj, dict):
                bbox = obj.get("bbox")
                cls_name = obj.get("class_name", "vehicle")
                conf = obj.get("conf", 0.0)
                track_id = obj.get("track_id", hash(tuple(bbox)) % 10000)
            else:
                continue

            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            y_bottom = y2
            y_top = y1

            zone = self.config.get_zone_name(y_bottom, y_top)

            current_track_ids.add(track_id)

            # Tạo mới hoặc cập nhật state
            if track_id not in self.tracked_vehicles:
                self.tracked_vehicles[track_id] = VehicleZoneState(track_id, zone)

            self.tracked_vehicles[track_id].update(zone, y_bottom, self.frame_count)

            if self.debug:
                print(f"  Track {track_id}: zone={zone}, y_bottom={y_bottom}")

        # Dọn dẹp các track không còn hoạt động
        inactive_ids = [
            tid for tid, state in self.tracked_vehicles.items()
            if not state.should_be_tracked()
        ]
        for tid in inactive_ids:
            del self.tracked_vehicles[tid]

        # Kiểm tra vi phạm khi đèn đỏ
        if self.red_light_active:
            for tid, state in self.tracked_vehicles.items():
                if state.violation_detected:
                    continue

                # Bỏ qua xe đang thoát giao lộ
                if state.is_clearing_intersection():
                    continue

                if state.has_violated_red_light():
                    # Tìm thông tin xe
                    obj_info = None
                    for obj in road_users_or_tracked:
                        if isinstance(obj, dict):
                            if obj.get("track_id") == tid:
                                obj_info = obj
                                break

                    violation = {
                        "bbox": obj_info.get("bbox") if obj_info else None,
                        "class_name": obj_info.get("class_name", "vehicle") if obj_info else "vehicle",
                        "conf": obj_info.get("conf", 0.0) if obj_info else 0.0,
                        "details": "Vượt đèn đỏ (zone-based)",
                        "track_id": tid,
                        "zone_history": state.zones_visited.copy(),
                        "violation_type": "RED_LIGHT_VIOLATION",
                    }

                    # Fallback: tìm bbox từ lịch sử
                    if violation["bbox"] is None and state.positions_history:
                        violation["bbox"] = (0, 0, 0, 0)

                    violations.append(violation)
                    state.violation_detected = True

        return violations

    def process_detected_objects(self, lights: List[Dict], road_users: List[Dict]) -> List[Dict]:
        """
        Phương thức tích hợp: nhận đầu ra từ detect_traffic_scene(),
        tự động cập nhật trạng thái đèn và phát hiện vi phạm.

        Args:
            lights: list đèn từ detect_traffic_scene()
            road_users: list người/phương tiện từ detect_traffic_scene()

        Returns:
            list vi phạm
        """
        self.update_from_traffic_lights(lights)
        return self.process_vehicles(road_users)

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ các vùng lên frame để debug/visualization."""
        h, w = frame.shape[:2]

        # Màu sắc các vùng (BGR với alpha)
        colors = {
            "waiting": (200, 200, 100),      # Vàng nhạt
            "stop": (100, 100, 200),         # Đỏ nhạt
            "intersection": (100, 200, 100), # Xanh lá nhạt
        }

        # Vẽ từng vùng
        overlay = frame.copy()

        if self.config.direction == "up":
            # Rear-view: xe đi từ dưới lên (Y lớn -> Y nhỏ)
            # Waiting zone: gần camera (Y lớn)
            # Intersection: xa camera (Y nhỏ)
            cv2.rectangle(overlay,
                          (0, self.config.waiting_end),
                          (w, self.config.waiting_start),
                          colors["waiting"], -1)

            cv2.rectangle(overlay,
                          (0, self.config.stop_end),
                          (w, self.config.stop_start),
                          colors["stop"], -1)

            cv2.rectangle(overlay,
                          (0, self.config.intersection_end),
                          (w, self.config.intersection_start),
                          colors["intersection"], -1)
        else:
            # Top-down: xe đi từ trên xuống (Y nhỏ -> Y lớn)
            cv2.rectangle(overlay,
                          (0, self.config.waiting_start),
                          (w, self.config.waiting_end),
                          colors["waiting"], -1)

            cv2.rectangle(overlay,
                          (0, self.config.stop_start),
                          (w, self.config.stop_end),
                          colors["stop"], -1)

            cv2.rectangle(overlay,
                          (0, self.config.intersection_start),
                          (w, self.config.intersection_end),
                          colors["intersection"], -1)

        # Blend overlay
        alpha = 0.15
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Vẽ đường viền và nhãn
        if self.config.direction == "up":
            y_centers = {
                "WAITING ZONE": (self.config.waiting_start + self.config.waiting_end) // 2,
                "STOP ZONE": (self.config.stop_start + self.config.stop_end) // 2,
                "INTERSECTION": (self.config.intersection_start + self.config.intersection_end) // 2,
            }
        else:
            y_centers = {
                "WAITING ZONE": (self.config.waiting_start + self.config.waiting_end) // 4,
                "STOP ZONE": (self.config.stop_start + self.config.stop_end) // 2,
                "INTERSECTION": (self.config.intersection_start + self.config.intersection_end) // 2,
            }

        for label, y_center in y_centers.items():
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            x_center = (w - text_w) // 2
            cv2.putText(frame, label, (x_center, y_center),
                        font, font_scale, (255, 255, 255), thickness)

        # Vẽ vạch dừng (stop line)
        stop_line_y = self.config.stop_line_y
        cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 0, 255), 3)
        cv2.putText(frame, "STOP LINE", (10, stop_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def reset(self):
        """Reset toàn bộ trạng thái."""
        self.tracked_vehicles.clear()
        self.frame_count = 0
        self.red_light_active = False
        self.red_light_start_frame = -1

    def get_stats(self) -> Dict:
        """Lấy thống kê hiện tại."""
        return {
            "frame_count": self.frame_count,
            "red_light_active": self.red_light_active,
            "red_light_duration_frames": (
                self.frame_count - self.red_light_start_frame
                if self.red_light_active and self.red_light_start_frame >= 0
                else 0
            ),
            "tracked_vehicles": len(self.tracked_vehicles),
            "violations_detected": sum(
                1 for s in self.tracked_vehicles.values() if s.violation_detected
            ),
            "zone_config": {
                "waiting_end": self.config.waiting_end,
                "stop_start": self.config.stop_start,
                "stop_end": self.config.stop_end,
                "intersection_start": self.config.intersection_start,
                "intersection_end": self.config.intersection_end,
            }
        }