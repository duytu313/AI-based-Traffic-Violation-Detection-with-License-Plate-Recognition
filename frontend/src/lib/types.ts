export interface Vehicle {
  bbox: number[];
  vtype: string;
  color: string;
  conf: number;
}

export interface MatchedPlate {
  bbox: number[];
  vtype: string;
  color: string;
  plate_text: string;
}

export interface Violation {
  type: string;
  details: string;
  bbox: number[];
  conf: number;
}

export interface ProcessImageResponse {
  vehicles: Vehicle[];
  matched_plates: MatchedPlate[];
  violations: ViolationGroup[];
  red_light_violations: any[];
  image_base64: string;
  stats: {
    total_vehicles: number;
    total_violations: number;
    total_red_light: number;
  };
}

export interface ViolationGroup {
  bbox: number[];
  vtype: string;
  color: string;
  plate_text: string;
  violations: Violation[];
}

export interface DBVehicle {
  id: number;
  track_id: number | null;
  license_plate: string | null;
  vehicle_type: string | null;
  color: string | null;
  entry_time: string;
  exit_time: string | null;
  fraud_alert: number;
  fraud_reason: string | null;
}

export interface DBViolation {
  id: number;
  vehicle_id: number | null;
  violation_type: string;
  timestamp: string;
  details: string;
  image_path: string | null;
  license_plate: string | null;
  vehicle_type: string | null;
  color: string | null;
}

export interface SpeedViolation {
  id: number;
  vehicle_id: number | null;
  track_id: number;
  license_plate: string;
  vehicle_type: string;
  color: string;
  speed_kmh: number;
  speed_limit: number;
  timestamp: string;
  image_path: string | null;
  entry_time: string | null;
}

export interface SpeedViolationStats {
  total_speed_violations: number;
  avg_speed: number;
  max_speed: number;
}

export interface Stats {
  total_vehicles: number;
  fraud_alerts: number;
  total_violations: number;
  total_speed_violations?: number;
}

export interface Config {
  enable_violation_detection: boolean;
  enable_red_light_detection: boolean;
  enable_bev_detection: boolean;
  enable_speed_violation_detection: boolean;
  violation_conf_limit: number;
  conf_more_than_two: number;
  conf_no_helmet: number;
  conf_using_mobile: number;
  traffic_light_conf: number;
  speed_limit: number;
  show_zones: boolean;
  show_bev: boolean;
  camera_direction: "down" | "up";
}

// Zone-based red light detection types
export interface ZoneConfig {
  direction: "down" | "up";
  waiting_end: number;
  stop_start: number;
  stop_end: number;
  intersection_start: number;
  intersection_end: number;
}

export interface ZoneStats {
  frame_count: number;
  red_light_active: boolean;
  red_light_duration_frames: number;
  tracked_vehicles: number;
  violations_detected: number;
  zone_config: {
    waiting_end: number;
    stop_start: number;
    stop_end: number;
    intersection_start: number;
    intersection_end: number;
  };
}

export interface RedLightViolationZone {
  bbox: number[];
  class_name: string;
  conf: number;
  details: string;
  track_id?: number;
  zone_history?: string[];
  zone_based?: boolean;
}

// ROI types
export interface ROIPoint {
  x: number;
  y: number;
}

export interface ROIConfig {
  points: ROIPoint[];
  name: string;
}
