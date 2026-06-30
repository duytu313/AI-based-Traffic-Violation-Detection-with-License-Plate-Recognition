"""
ROI (Region of Interest) Detector
Detects vehicles inside manually drawn polygon zones.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ROIConfig:
    """ROI zone configuration in polygon format."""
    points: List[Dict[str, int]]  # [{"x": 100, "y": 200}, ...]
    name: str = "violation_zone"

    def to_numpy(self) -> np.ndarray:
        """Convert points to numpy array for OpenCV."""
        return np.array([[p["x"], p["y"]] for p in self.points], dtype=np.int32)


class ROIDetector:
    """
    Detects whether a vehicle is inside the ROI zone.
    Uses point-in-polygon test.
    """

    def __init__(self, roi_config: Optional[ROIConfig] = None):
        self.config = roi_config
        self.violation_history: Dict[int, List[bool]] = {}  # track_id -> list of in_roi flags

    def set_roi(self, points: List[Dict[str, int]], name: str = "violation_zone"):
        """Update ROI zone."""
        self.config = ROIConfig(points=points, name=name)

    def clear_roi(self):
        """Clear ROI zone."""
        self.config = None

    def is_point_in_polygon(self, x: float, y: float) -> bool:
        """
        Check if point is inside polygon.
        Uses ray casting algorithm.
        """
        if self.config is None or len(self.config.points) < 3:
            return False

        polygon = self.config.to_numpy()
        point = (x, y)

        # Use OpenCV pointPolygonTest
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0

    def is_bbox_in_roi(self, bbox: tuple, threshold: float = 0.3) -> bool:
        """
        Check if vehicle (bbox) is inside ROI.
        Checks multiple points on bbox (center, corners, midpoints).
        
        Args:
            bbox: (x1, y1, x2, y2)
            threshold: minimum ratio of points inside ROI to consider as inside zone
            
        Returns:
            True if enough points are inside ROI
        """
        if self.config is None or len(self.config.points) < 3:
            return False

        x1, y1, x2, y2 = bbox
        
        # Check multiple points on bbox
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
        Process list of vehicles and detect ROI violations.
        
        Args:
            vehicles: list of dicts with keys: bbox, track_id, class_name, conf
            red_light_active: True if red light is active
            min_history: number of consecutive frames vehicle must be in ROI to consider as violation
                         (1 for single image, 2 for video to reduce false positive)
            
        Returns:
            list of detected violations
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

            # Save history
            if track_id not in self.violation_history:
                self.violation_history[track_id] = []
            self.violation_history[track_id].append(in_roi)

            # Detect violation: vehicle enters ROI when red light
            if red_light_active and in_roi:
                history = self.violation_history[track_id]
                if len(history) >= min_history and sum(history[-min_history:]) >= min_history:
                    violations.append({
                        "bbox": bbox,
                        "track_id": track_id,
                        "class_name": vehicle.get("class_name", "vehicle"),
                        "conf": vehicle.get("conf", 0.0),
                        "details": "Red light running (ROI)",
                        "violation_type": "RED_LIGHT_VIOLATION",
                    })

        # Clean up old history
        self._cleanup_history()

        return violations

    def _cleanup_history(self, max_history: int = 100):
        """Clean up old history."""
        for track_id in list(self.violation_history.keys()):
            if len(self.violation_history[track_id]) > max_history:
                self.violation_history[track_id] = self.violation_history[track_id][-max_history:]

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI zone on frame."""
        if self.config is None or len(self.config.points) < 3:
            return frame

        polygon = self.config.to_numpy()

        # Draw polygon fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (0, 255, 255, 50))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw border
        cv2.polylines(frame, [polygon], True, (0, 255, 255), 3)

        # Draw zone name
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
        """Get current ROI configuration."""
        if self.config is None:
            return None
        return {
            "points": self.config.points,
            "name": self.config.name,
        }

    def reset(self):
        """Reset state."""
        self.violation_history.clear()