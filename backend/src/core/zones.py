"""
Red Light Zone Detector
Detects red light violations based on zones (zone-based) instead of lines (line-based).

Zone diagram:

      Camera
┌─────────────────┐
│  Waiting Zone   │  - Waiting area (before stop line)
├─────────────────┤
│   Stop Zone     │  - Stop line area (critical zone)
├─────────────────┤
│ Intersection    │  - Intersection area (past stop line)
└─────────────────┘

Violation detection logic:
  - When red light:
    * Vehicle goes Waiting → Stop → Intersection => VIOLATION
    * Vehicle stops in Waiting => OK (waiting correctly)
    * Vehicle already in Intersection when light turns red => OK (exiting intersection)
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ZoneConfig:
    """
    Zone configuration based on Y coordinates (along vehicle movement direction).
    
    direction: camera direction
      - "down": camera looks from top-down, vehicle moves from small Y to large Y
      - "up": camera looks from rear-view, vehicle moves from large Y to small Y
    """
    direction: str = "down"       # "down" = top-down, "up" = rear-view
    waiting_start: int = 0        # Y start of waiting zone
    waiting_end: int = 200        # Y end of waiting zone
    stop_start: int = 200         # Y start of stop zone
    stop_end: int = 300           # Y end of stop zone
    intersection_start: int = 300 # Y start of intersection zone
    intersection_end: int = 500   # Y end of intersection zone

    @property
    def stop_line_y(self) -> int:
        """Stop line position (center of stop zone)."""
        return (self.stop_start + self.stop_end) // 2

    def get_zone_name(self, y_bottom: int, y_top: int = None) -> str:
        """
        Determine zone name based on Y coordinates and camera direction.
        
        With direction="down" (top-down): vehicle moves from top to bottom
          - waiting: y_bottom <= waiting_end
          - stop: waiting_end < y_bottom <= stop_end
          - intersection: y_bottom > intersection_start
          
        With direction="up" (rear-view): vehicle moves from bottom to top
          - waiting: y_bottom >= waiting_end (near camera)
          - stop: waiting_end > y_bottom >= stop_start
          - intersection: y_bottom < intersection_start (far from camera)
        """
        if self.direction == "up":
            # Reverse: camera looks from behind
            if y_bottom >= self.waiting_end:
                return "waiting"
            elif y_bottom >= self.stop_start:
                return "stop"
            elif y_bottom <= self.intersection_start:
                return "intersection"
        else:
            # Default: camera looks from top-down
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
    Tracks zone state of a vehicle across frames.
    """
    def __init__(self, track_id: int, initial_zone: str):
        self.track_id = track_id
        self.zones_visited = [initial_zone]  # History of zones visited
        self.current_zone = initial_zone
        self.first_seen_time = cv2.getTickCount()
        self.last_seen_time = self.first_seen_time
        self.violation_detected = False
        self.violation_frame = None
        self.positions_history: List[Tuple[int, int]] = []  # (y_bottom, frame_count)

    def update(self, zone: str, y_bottom: int, frame_count: int = 0):
        """Update vehicle's new position."""
        self.last_seen_time = cv2.getTickCount()
        self.positions_history.append((y_bottom, frame_count))

        if zone != self.current_zone:
            if zone not in self.zones_visited:
                self.zones_visited.append(zone)
            self.current_zone = zone

    def has_violated_red_light(self) -> bool:
        """
        Check violation based on zone history.
        Violation when: waiting → stop → intersection (passes through all 3 zones).
        """
        if self.violation_detected:
            return True

        # Check order: must have waiting, stop, intersection
        has_waiting = "waiting" in self.zones_visited
        has_stop = "stop" in self.zones_visited
        has_intersection = "intersection" in self.zones_visited

        if has_waiting and has_stop and has_intersection:
            # Check order of appearance (excluding "unknown")
            ordered_zones = [z for z in self.zones_visited if z != "unknown"]
            if len(ordered_zones) >= 3:
                # Find index of each zone
                try:
                    idx_waiting = ordered_zones.index("waiting")
                    idx_stop = ordered_zones.index("stop")
                    idx_intersection = ordered_zones.index("intersection")
                    # Must follow order: waiting -> stop -> intersection
                    if idx_waiting < idx_stop < idx_intersection:
                        self.violation_detected = True
                        return True
                except ValueError:
                    pass

        # Case: vehicle appears in waiting then goes straight to intersection
        # (speeding through red light)
        if has_waiting and has_intersection and not has_stop:
            # Check if it went from waiting directly to intersection (speeding through)
            ordered_zones = [z for z in self.zones_visited if z != "unknown"]
            if len(ordered_zones) >= 2:
                try:
                    idx_waiting = ordered_zones.index("waiting")
                    idx_intersection = ordered_zones.index("intersection")
                    if idx_waiting < idx_intersection:
                        # Check if vehicle passed through stop zone very quickly
                        self.violation_detected = True
                        return True
                except ValueError:
                    pass

        return False

    def is_clearing_intersection(self) -> bool:
        """
        Vehicle is exiting intersection (was in intersection from the start).
        Not counted as violation.
        """
        if len(self.zones_visited) == 1 and self.zones_visited[0] == "intersection":
            return True
        return False

    def should_be_tracked(self, max_inactive_ticks: int = 5000000) -> bool:
        """Check if vehicle is still being tracked."""
        elapsed = cv2.getTickCount() - self.last_seen_time
        return elapsed < max_inactive_ticks


class RedLightZoneDetector:
    """
    Detects red light violations based on zones (zone-based).
    Uses ZoneConfig and VehicleZoneState to track and detect violations.
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
        """Update zone configuration."""
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
        """Update red light status."""
        if is_red and not self.red_light_active:
            self.red_light_start_frame = self.frame_count
        self.red_light_active = is_red

    def update_from_traffic_lights(self, lights: List[Dict]):
        """Update red light status from traffic light list."""
        is_red = any(light.get("state") == "red" for light in lights)
        self.set_red_light(is_red)

    def process_vehicles(self, road_users_or_tracked: List[Any]) -> List[Dict]:
        """
        Process vehicle list from traffic light detection or tracker.
        Returns list of detected violations.

        Args:
            road_users_or_tracked: list of dicts with keys: bbox, class_name, conf
                                   or list of tracks with bbox

        Returns:
            list of dicts: {bbox, class_name, conf, details, zone_history}
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

            # Create new or update state
            if track_id not in self.tracked_vehicles:
                self.tracked_vehicles[track_id] = VehicleZoneState(track_id, zone)

            self.tracked_vehicles[track_id].update(zone, y_bottom, self.frame_count)

            if self.debug:
                print(f"  Track {track_id}: zone={zone}, y_bottom={y_bottom}")

        # Clean up inactive tracks
        inactive_ids = [
            tid for tid, state in self.tracked_vehicles.items()
            if not state.should_be_tracked()
        ]
        for tid in inactive_ids:
            del self.tracked_vehicles[tid]

        # Check violations when red light
        if self.red_light_active:
            for tid, state in self.tracked_vehicles.items():
                if state.violation_detected:
                    continue

                # Skip vehicles exiting intersection
                if state.is_clearing_intersection():
                    continue

                if state.has_violated_red_light():
                    # Find vehicle information
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
                        "details": "Red light running (zone-based)",
                        "track_id": tid,
                        "zone_history": state.zones_visited.copy(),
                        "violation_type": "RED_LIGHT_VIOLATION",
                    }

                    # Fallback: find bbox from history
                    if violation["bbox"] is None and state.positions_history:
                        violation["bbox"] = (0, 0, 0, 0)

                    violations.append(violation)
                    state.violation_detected = True

        return violations

    def process_detected_objects(self, lights: List[Dict], road_users: List[Dict]) -> List[Dict]:
        """
        Integration method: receives output from detect_traffic_scene(),
        automatically updates light status and detects violations.

        Args:
            lights: list of lights from detect_traffic_scene()
            road_users: list of people/vehicles from detect_traffic_scene()

        Returns:
            list of violations
        """
        self.update_from_traffic_lights(lights)
        return self.process_vehicles(road_users)

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw zones on frame for debug/visualization."""
        h, w = frame.shape[:2]

        # Zone colors (BGR with alpha)
        colors = {
            "waiting": (200, 200, 100),      # Light yellow
            "stop": (100, 100, 200),         # Light red
            "intersection": (100, 200, 100), # Light green
        }

        # Draw each zone
        overlay = frame.copy()

        if self.config.direction == "up":
            # Rear-view: vehicle moves from bottom to top (large Y -> small Y)
            # Waiting zone: near camera (large Y)
            # Intersection: far from camera (small Y)
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
            # Top-down: vehicle moves from top to bottom (small Y -> large Y)
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

        # Draw borders and labels
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

        # Draw stop line
        stop_line_y = self.config.stop_line_y
        cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 0, 255), 3)
        cv2.putText(frame, "STOP LINE", (10, stop_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def reset(self):
        """Reset all state."""
        self.tracked_vehicles.clear()
        self.frame_count = 0
        self.red_light_active = False
        self.red_light_start_frame = -1

    def get_stats(self) -> Dict:
        """Get current statistics."""
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