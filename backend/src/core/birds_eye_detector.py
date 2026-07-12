"""
Bird's Eye View (BEV) Red Light Detector
Detects red light violations using Perspective Transform to 3D space.

Logic from Colab notebook:
  1. Detect red light from YOLO class name ("trafficLight" + "red")
  2. Buffer 30 frames to prevent red light flickering
  3. Convert vehicle bottom-center to BEV space
  4. If in trapezoid zone and y_3d < STOP_LINE_3D_Y → violation
  5. officially_violated_ids: keep red frame forever
  6. logged_violated_ids: log only once
  7. Draw: yellow trapezoid, camera stop line, VIOLATION/WAITING labels
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Set
from dataclasses import dataclass, field


@dataclass
class BEVConfig:
    """
    Configuration for Bird's Eye View detector.
    
    src_points: 4 points in original image space (trapezoid)
      [bottom-left, bottom-right, top-right, top-left]
    dst_points: 4 points in 3D BEV space (rectangle)
      [bottom-left, bottom-right, top-right, top-left]
    stop_line_3d_y: Stop line position in BEV space (Y coordinate)
    red_light_buffer_frames: Number of buffer frames for red light (anti-flicker)
    """
    src_points: np.ndarray = field(default_factory=lambda: np.zeros((4, 2), dtype=np.float32))
    dst_points: np.ndarray = field(default_factory=lambda: np.float32([
        [0, 600],      # bottom-left
        [400, 600],    # bottom-right
        [400, 0],      # top-right
        [0, 0]         # top-left
    ]))
    stop_line_3d_y: int = 400
    red_light_buffer_frames: int = 30


class BirdsEyeRedLightDetector:
    """
    Detects red light violations using Perspective Transform to BEV.
    Code converted from Colab notebook, keeping the same logic.
    
    Features:
      - Convert 2D coordinates to 3D BEV
      - Detect virtual stop line crossing in BEV space
      - Only report violation once per vehicle
      - Red light buffer to prevent flickering (30 frames)
      - Keep red frame permanently for violating vehicles
      - "WAITING" status when red light but not crossed stop line
    """

    def __init__(self, config: Optional[BEVConfig] = None):
        self.config = config or BEVConfig()
        
        # Perspective transform matrices - only compute when src_points is valid
        self.M = None
        self.M_inv = None
        if np.any(self.config.src_points != 0):
            try:
                self.M = cv2.getPerspectiveTransform(self.config.src_points, self.config.dst_points)
                self.M_inv = np.linalg.inv(self.M)
            except Exception:
                self.M = None
                self.M_inv = None
        
        # Red light state with buffer (like Colab)
        self.red_light_counter = 0
        self.RED_LIGHT_BUFFER_FRAMES = self.config.red_light_buffer_frames
        
        # Violation tracking (like Colab)
        self.officially_violated_ids: Set[str] = set()  # Set of vehicles that HAVE BEEN PENALIZED
        self.logged_violated_ids: Set[str] = set()      # Set of vehicles that HAVE BEEN LOGGED
        
        # Frame counter
        self.frame_count = 0

    # ======================== CONFIGURATION ========================

    def set_src_points(self, points: np.ndarray):
        """Update 4 source points (trapezoid in image space)."""
        self.config.src_points = points.astype(np.float32)
        self._recompute_transform()

    def set_dst_points(self, points: np.ndarray):
        """Update 4 destination points (rectangle in BEV space)."""
        self.config.dst_points = points.astype(np.float32)
        self._recompute_transform()

    def set_stop_line_3d_y(self, y: int):
        """Update stop line position in BEV space."""
        self.config.stop_line_3d_y = y

    def set_red_light_buffer_frames(self, frames: int):
        """Update number of buffer frames for red light."""
        self.config.red_light_buffer_frames = frames
        self.RED_LIGHT_BUFFER_FRAMES = frames

    def _recompute_transform(self):
        """Recompute perspective transform matrix."""
        self.M = cv2.getPerspectiveTransform(self.config.src_points, self.config.dst_points)
        self.M_inv = np.linalg.inv(self.M)

    # ======================== PERSPECTIVE TRANSFORM (like Colab) ========================

    def convert_to_3d_point(self, cx: float, cy: float, transform_matrix=None) -> Tuple[float, float]:
        """
        Convert 2D coordinates (image space) to 3D coordinates (BEV space).
        Same as convert_to_3d_point function in Colab.
        
        Args:
            cx: X coordinate in image space
            cy: Y coordinate in image space
            transform_matrix: Perspective transform matrix (M), defaults to self.M
            
        Returns:
            (x_3d, y_3d) in BEV space
        """
        if transform_matrix is None:
            transform_matrix = self.M
        if transform_matrix is None:
            return (0.0, 0.0)  # BEV not configured yet
        point = np.array([[[cx, cy]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, transform_matrix)
        return transformed_point[0][0]

    def get_stop_line_2d(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Calculate stop line in image space from BEV stop line.
        Like Colab: use M_inv to project 3D stop line back to 2D.
        
        Returns:
            ((left_x, left_y), (right_x, right_y)) in image space
        """
        line_3d_pts = np.array([
            [[0, self.config.stop_line_3d_y]],
            [[400, self.config.stop_line_3d_y]]
        ], dtype=np.float32)
        line_2d_pts = cv2.perspectiveTransform(line_3d_pts, self.M_inv)
        left = (int(line_2d_pts[0][0][0]), int(line_2d_pts[0][0][1]))
        right = (int(line_2d_pts[1][0][0]), int(line_2d_pts[1][0][1]))
        return left, right

    def is_point_in_bev_zone(self, x_3d: float, y_3d: float) -> bool:
        """
        Check if point is within valid BEV zone.
        Like Colab: (0 <= x_3d <= 400) and (0 <= y_3d <= 600)
        """
        return (0 <= x_3d <= 400) and (0 <= y_3d <= 600)

    # ======================== RED LIGHT STATE (like Colab) ========================

    def update_red_light(self, lights: List[Dict]):
        """
        Update red light status from traffic light list.
        Like Colab: check class name has "trafficLight" and "red".
        Also uses analyzed state from infer_traffic_light_state().
        
        Update red_light_counter with buffer to prevent flickering.
        """
        # Detect red light from YOLO class names (like Colab)
        yolo_detected_red = False
        for light in lights:
            cls_name = light.get("class_name", "").lower()
            state = light.get("state", "").lower()
            # Check both YOLO class name AND analyzed state
            if ("trafficlight" in cls_name and "red" in cls_name) or state == "red":
                yolo_detected_red = True
                break

        if yolo_detected_red:
            self.red_light_counter = self.RED_LIGHT_BUFFER_FRAMES
        else:
            if self.red_light_counter > 0:
                self.red_light_counter -= 1

    @property
    def is_red_light_active(self) -> bool:
        """Red light status after buffer. Like Colab: counter > 0."""
        return self.red_light_counter > 0

    # ======================== VEHICLE PROCESSING (like Colab) ========================

    def process_vehicle(
        self,
        track_id: int,
        class_name: str,
        bbox: Tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        """
        Process a vehicle and check red light violation.
        Exactly same logic as Colab:
        
          1. Calculate center_x and bottom_y (cx, cy_tail)
          2. Convert to BEV (x_3d, y_3d)
          3. Check in zone and past line
          4. If red light + past line → VIOLATION
          5. officially_violated_ids: add to blacklist
          6. logged_violated_ids: log only once
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)  # Like Colab: center x
        cy_tail = y2              # Like Colab: bottom y

        # Convert coordinates to 3D Bird's Eye View (like Colab)
        x_3d, y_3d = self.convert_to_3d_point(cx, cy_tail, self.M)

        unique_key = f"{class_name}_{track_id}"

        # Check if vehicle is inside trapezoid zone (like Colab)
        is_inside_zone = self.is_point_in_bev_zone(x_3d, y_3d)
        is_past_line_3d = is_inside_zone and (y_3d < self.config.stop_line_3d_y)

        # Default result (green)
        color = (0, 255, 0)
        label = unique_key
        is_violation_bool = False
        is_first_violation_bool = False
        show_waiting_label = False

        # If this vehicle HAS BEEN CAUGHT violating before (like Colab)
        if unique_key in self.officially_violated_ids:
            is_violation_bool = True

        # If red light and vehicle just crossed stop line for first time (like Colab)
        elif self.is_red_light_active and is_past_line_3d:
            # Add to blacklist to keep red frame permanently (like Colab)
            self.officially_violated_ids.add(unique_key)
            is_violation_bool = True

            # CHECK: Only log once for this ID (like Colab)
            if unique_key not in self.logged_violated_ids:
                self.logged_violated_ids.add(unique_key)
                is_first_violation_bool = True

        # Red light but vehicle hasn't crossed stop line (like Colab: WAITING)
        elif self.is_red_light_active:
            show_waiting_label = True

        # Green/yellow light normal (like Colab: green)
        # default color is green, label is unique_key

        result = {
            "unique_key": unique_key,
            "is_inside_zone": is_inside_zone,
            "is_past_line_3d": is_past_line_3d,
            "bev_coords": (float(x_3d), float(y_3d)),
            "is_violation": is_violation_bool,
            "first_time_violation": is_first_violation_bool,
            "show_waiting_label": show_waiting_label,
            "color": color,
            "label": label,
        }

        return result

    def reset_frame(self):
        """Call at the start of each new frame (increment frame counter)."""
        self.frame_count += 1

    def reset(self):
        """Reset all state to initial."""
        self.red_light_counter = 0
        self.officially_violated_ids.clear()
        self.logged_violated_ids.clear()
        self.frame_count = 0

    # ======================== DRAWING (like Colab) ========================

    def draw_all(
        self,
        frame: np.ndarray,
        tracked_vehicles: List[Tuple[int, Tuple[int,int,int,int], int, float]],
        vehicle_classes: Dict[int, str],
    ) -> np.ndarray:
        """
        Draw all BEV overlays on frame (like Colab):
          1. Control trapezoid zone (yellow)
          2. Stop line (orange)
          3. Label for each vehicle (VIOLATION / WAITING / green)
          4. System status
        """
        # 1. Draw control trapezoid zone - only draw when points are configured (not default)
        pts = self.config.src_points.astype(np.int32)
        if np.any(pts != 0):  # Only draw if configured
            # Only draw outline, no fill
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)  # Only draw red outline

            # 2. Draw virtual stop line
            try:
                left, right = self.get_stop_line_2d()
                cv2.line(frame, left, right, (0, 152, 255), 2)
            except Exception:
                pass

        # 3. Process each vehicle (like Colab)
        for trk_id, bbox, cls_id, conf in tracked_vehicles:
            x1, y1, x2, y2 = bbox
            class_name = vehicle_classes.get(cls_id, "vehicle")
            
            # Calculate BEV for this vehicle
            cx = int((x1 + x2) / 2)
            cy_tail = y2
            x_3d, y_3d = self.convert_to_3d_point(cx, cy_tail, self.M)
            unique_key = f"{class_name}_{trk_id}"
            is_inside_zone = self.is_point_in_bev_zone(x_3d, y_3d)
            is_past_line_3d = is_inside_zone and (y_3d < self.config.stop_line_3d_y)

            color = (0, 255, 0)  # Default green
            label = unique_key

            # If has violated before (like Colab)
            if unique_key in self.officially_violated_ids:
                color = (0, 0, 255)
                label = f"VIOLATION 3D! {unique_key.upper()}"
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + 180, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Red light and not crossed stop line (like Colab: WAITING)
            elif self.is_red_light_active and is_inside_zone and not is_past_line_3d:
                color = (0, 255, 255)
                label = f"WAITING: {unique_key}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # Normal green/yellow light
            else:
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 4. Display system status (like Colab)
        status_text = "3D SYSTEM: RED LIGHT" if self.is_red_light_active else "3D SYSTEM: GREEN/YELLOW LIGHT"
        status_color = (0, 0, 255) if self.is_red_light_active else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return frame

    # ======================== STATS ========================

    def get_stats(self) -> Dict:
        """Get statistics information."""
        return {
            "frame_count": self.frame_count,
            "red_light_active": self.is_red_light_active,
            "red_light_counter": self.red_light_counter,
            "total_violations": len(self.officially_violated_ids),
            "total_logged": len(self.logged_violated_ids),
            "src_points": self.config.src_points.tolist(),
            "dst_points": self.config.dst_points.tolist(),
            "stop_line_3d_y": self.config.stop_line_3d_y,
        }