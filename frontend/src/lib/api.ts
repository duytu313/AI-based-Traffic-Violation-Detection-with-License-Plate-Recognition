import axios from "axios";
import type { Config, ProcessImageResponse, DBVehicle, DBViolation, Stats } from "./types";

// Use relative URL so Next.js rewrites proxy API requests to backend
// If NEXT_PUBLIC_API_URL is set, use it directly (for production/deployment)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await api.get("/api/health");
    return res.data.status === "ok";
  } catch {
    return false;
  }
}

export async function processImage(file: File, config: Config): Promise<ProcessImageResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("enable_violation_detection", String(config.enable_violation_detection));
  formData.append("enable_red_light_detection", String(config.enable_red_light_detection));
  formData.append("enable_bev_detection", String(config.enable_bev_detection));
  formData.append("violation_conf_limit", String(config.violation_conf_limit));
  formData.append("conf_more_than_two", String(config.conf_more_than_two));
  formData.append("conf_no_helmet", String(config.conf_no_helmet));
  formData.append("conf_using_mobile", String(config.conf_using_mobile));
  formData.append("traffic_light_conf", String(config.traffic_light_conf));
  formData.append("show_zones", String(config.show_zones));
  formData.append("show_bev", String(config.show_bev));
  formData.append("camera_direction", config.camera_direction);
  const res = await api.post("/api/process-image", formData);
  return res.data;
}

export async function startWebcam(source: string, backend: string): Promise<any> {
  const formData = new FormData();
  formData.append("source", source);
  formData.append("backend", backend);
  const res = await api.post("/api/webcam/start", formData);
  return res.data;
}

export async function stopWebcam(): Promise<any> {
  const res = await api.post("/api/webcam/stop");
  return res.data;
}

export async function getWebcamFrame(): Promise<Blob> {
  const res = await api.get("/api/webcam/frame", { responseType: "blob" });
  return res.data;
}

export async function getWebcamStatus(): Promise<any> {
  const res = await api.get("/api/webcam/status");
  return res.data;
}

export async function getWebcamDetected(): Promise<any> {
  const res = await api.get("/api/webcam/detected");
  return res.data;
}

export async function getVehicles(params?: {
  limit?: number;
  offset?: number;
  license_plate?: string;
  start_time?: string;
  end_time?: string;
}): Promise<DBVehicle[]> {
  const res = await api.get("/api/vehicles", { params });
  return res.data;
}

export async function getViolations(params?: {
  limit?: number;
  offset?: number;
  violation_type?: string;
}): Promise<DBViolation[]> {
  const res = await api.get("/api/violations", { params });
  return res.data;
}

export async function getStats(): Promise<Stats> {
  const res = await api.get("/api/stats");
  return res.data;
}

export async function getFraudAlerts(limit?: number): Promise<any[]> {
  const res = await api.get("/api/fraud", { params: { limit } });
  return res.data;
}

// ======================== SPEED VIOLATION API ========================

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

export async function getSpeedViolations(limit = 50, licensePlate?: string): Promise<SpeedViolation[]> {
  const res = await api.get("/api/speed-violations", { params: { limit, license_plate: licensePlate } });
  return res.data;
}

export async function getSpeedViolationStats(): Promise<SpeedViolationStats> {
  const res = await api.get("/api/speed-violation-stats");
  return res.data;
}

export async function setSpeedLimit(speedLimit: number): Promise<{ status: string; speed_limit: number }> {
  const formData = new FormData();
  formData.append("speed_limit", String(speedLimit));
  const res = await api.post("/api/speed-limit", formData);
  return res.data;
}

export async function getSpeedLimit(): Promise<{ speed_limit: number }> {
  const res = await api.get("/api/speed-limit");
  return res.data;
}

export async function processVideo(file: File, config: any): Promise<any> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("enable_violation_detection", String(config.enable_violation_detection));
  formData.append("enable_red_light_detection", String(config.enable_red_light_detection));
  formData.append("violation_conf_limit", String(config.violation_conf_limit));
  formData.append("conf_more_than_two", String(config.conf_more_than_two));
  formData.append("conf_no_helmet", String(config.conf_no_helmet));
  formData.append("conf_using_mobile", String(config.conf_using_mobile));
  formData.append("traffic_light_conf", String(config.traffic_light_conf));
  formData.append("max_frames", "100");
  formData.append("speed_limit", String(config.speed_limit || 60));
  const res = await api.post("/api/process-video", formData);
  return res.data;
}

export async function startVideoStream(file: File, config: any): Promise<any> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("enable_violation_detection", String(config.enable_violation_detection));
  formData.append("enable_red_light_detection", String(config.enable_red_light_detection));
  formData.append("violation_conf_limit", String(config.violation_conf_limit));
  formData.append("conf_more_than_two", String(config.conf_more_than_two));
  formData.append("conf_no_helmet", String(config.conf_no_helmet));
  formData.append("conf_using_mobile", String(config.conf_using_mobile));
  formData.append("traffic_light_conf", String(config.traffic_light_conf));
  const res = await api.post("/api/video/stream", formData);
  return res.data;
}

export async function stopVideoStream(): Promise<any> {
  const res = await api.post("/api/video/stop");
  return res.data;
}

export async function getVideoFrame(): Promise<Blob> {
  const res = await api.get("/api/video/frame", { responseType: "blob" });
  return res.data;
}

export async function getVideoStatus(): Promise<any> {
  const res = await api.get("/api/video/status");
  return res.data;
}

export async function getVideoDetected(): Promise<any> {
  const res = await api.get("/api/video/detected");
  return res.data;
}


export type ParkingGate = "entry" | "exit";

export interface ParkingGateStatus {
  gate: ParkingGate;
  running: boolean;
  source: string | null;
  fps: number;
  detected: number;
  violations: number;
  latest_plate: string;
  latest_info: string;
}

export interface ParkingGateDetectedVehicle {
  vehicle_b64: string | null;
  plate_b64: string | null;
  plate_text: string;
  info: string;
}

export interface ParkingGateDetectedResponse {
  gate: ParkingGate;
  vehicles: ParkingGateDetectedVehicle[];
  latest: ParkingGateDetectedVehicle | null;
}

export async function startParkingGate(gate: ParkingGate, source: string, backend = "AUTO"): Promise<{ status: string; gate: string; source: string }> {
  const formData = new FormData();
  formData.append("source", source);
  formData.append("backend", backend);
  const res = await api.post(`/api/parking/${gate}/start`, formData);
  return res.data;
}

export async function stopParkingGate(gate: ParkingGate): Promise<{ status: string; gate: string }> {
  const res = await api.post(`/api/parking/${gate}/stop`);
  return res.data;
}

export async function getParkingGateFrame(gate: ParkingGate): Promise<Blob> {
  const res = await api.get(`/api/parking/${gate}/frame`, { responseType: "blob" });
  return res.data;
}

export async function getParkingGateStatus(gate: ParkingGate): Promise<ParkingGateStatus> {
  const res = await api.get(`/api/parking/${gate}/status`);
  return res.data;
}

export async function getParkingGateDetected(gate: ParkingGate): Promise<ParkingGateDetectedResponse> {
  const res = await api.get(`/api/parking/${gate}/detected`);
  return res.data;
}

// ======================== LOGISTICS CAMERA API ========================
export type LogisticsCamera = "gate" | "construction_site";

export interface LogisticsCameraStatus {
  camera: LogisticsCamera;
  running: boolean;
  source: string | null;
  fps: number;
  detected: number;
  violations: number;
  unknown_vehicle_alerts: number;
  latest_plate: string;
  latest_info: string;
}

export interface LogisticsDetectedVehicle {
  vehicle_b64: string | null;
  plate_b64: string | null;
  plate_text: string;
  info: string;
}

export interface LogisticsDetectedResponse {
  camera: LogisticsCamera;
  vehicles: LogisticsDetectedVehicle[];
  latest: LogisticsDetectedVehicle | null;
  unknown_vehicle_alerts: Array<{
    time: number;
    track_id: number;
    vehicle_type: string;
    color: string;
    message: string;
  }>;
}

export async function startLogisticsCamera(camera: LogisticsCamera, source: string, backend = "AUTO"): Promise<{ status: string; camera: string; source: string }> {
  const formData = new FormData();
  formData.append("source", source);
  formData.append("backend", backend);
  const res = await api.post(`/api/logistics/${camera}/start`, formData);
  return res.data;
}

export async function stopLogisticsCamera(camera: LogisticsCamera): Promise<{ status: string; camera: string }> {
  const res = await api.post(`/api/logistics/${camera}/stop`);
  return res.data;
}

export async function getLogisticsCameraFrame(camera: LogisticsCamera): Promise<Blob> {
  const res = await api.get(`/api/logistics/${camera}/frame`, { responseType: "blob" });
  return res.data;
}

export async function getLogisticsCameraStatus(camera: LogisticsCamera): Promise<LogisticsCameraStatus> {
  const res = await api.get(`/api/logistics/${camera}/status`);
  return res.data;
}

export async function getLogisticsCameraDetected(camera: LogisticsCamera): Promise<LogisticsDetectedResponse> {
  const res = await api.get(`/api/logistics/${camera}/detected`);
  return res.data;
}

// ======================== SMART CITY CAMERA API ========================
export type SmartCityCamera = "cam1" | "cam2" | "cam3" | "cam4";

export interface SmartCityCameraStatus {
  camera: SmartCityCamera;
  running: boolean;
  source: string | null;
  fps: number;
  detected: number;
  violations: number;
  latest_plate: string;
  latest_info: string;
}

export interface SmartCityDetectedVehicle {
  vehicle_b64: string | null;
  plate_b64: string | null;
  plate_text: string;
  info: string;
}

export interface SmartCityDetectedResponse {
  camera: SmartCityCamera;
  vehicles: SmartCityDetectedVehicle[];
  latest: SmartCityDetectedVehicle | null;
  count: number;
}

export async function startSmartCityCamera(camera: SmartCityCamera, source: string, backend = "AUTO"): Promise<{ status: string; camera: string; source: string }> {
  const formData = new FormData();
  formData.append("source", source);
  formData.append("backend", backend);
  const res = await api.post(`/api/smartcity/${camera}/start`, formData);
  return res.data;
}

export async function stopSmartCityCamera(camera: SmartCityCamera): Promise<{ status: string; camera: string }> {
  const res = await api.post(`/api/smartcity/${camera}/stop`);
  return res.data;
}

export async function getSmartCityCameraFrame(camera: SmartCityCamera): Promise<Blob> {
  const res = await api.get(`/api/smartcity/${camera}/frame`, { responseType: "blob" });
  return res.data;
}

export async function getSmartCityCameraStatus(camera: SmartCityCamera): Promise<SmartCityCameraStatus> {
  const res = await api.get(`/api/smartcity/${camera}/status`);
  return res.data;
}

export async function getSmartCityCameraDetected(camera: SmartCityCamera): Promise<SmartCityDetectedResponse> {
  const res = await api.get(`/api/smartcity/${camera}/detected`);
  return res.data;
}


// ======================== PARKING DATABASE API ========================

export interface ParkingSlot {
  id: string;
  zone: string;
  status: string;
  vehicle_type: string | null;
  current_plate: string | null;
  entry_time: string | null;
  fee: number;
}

export interface ParkingEntry {
  id: number;
  track_id: number | null;
  license_plate: string | null;
  vehicle_type: string | null;
  color: string | null;
  slot_id: string | null;
  zone: string | null;
  entry_time: string;
  exit_time: string | null;
  fee: number;
  status: string;
  fraud_alert: number;
  fraud_reason: string | null;
}

export interface ParkingStats {
  active: number;
  total: number;
  revenue: number;
  fraud: number;
}

export async function getParkingDbStats(): Promise<ParkingStats> {
  const res = await api.get("/api/parking/stats");
  return res.data;
}

export async function getParkingDbEntries(limit = 100, offset = 0): Promise<ParkingEntry[]> {
  const res = await api.get("/api/parking/entries", { params: { limit, offset } });
  return res.data;
}

export async function getParkingDbSlots(): Promise<ParkingSlot[]> {
  const res = await api.get("/api/parking/slots");
  return res.data;
}

// ======================== ROI API ========================

export interface ROIPoint {
  x: number;
  y: number;
}

export interface ROIConfig {
  points: ROIPoint[];
  name: string;
}

export async function setROI(points: ROIPoint[], name: string = "violation_zone"): Promise<{ status: string; roi_config: ROIConfig }> {
  const res = await api.post("/api/roi/set", { points, name });
  return res.data;
}

export async function getROI(): Promise<{ roi_config: ROIConfig | null }> {
  const res = await api.get("/api/roi/get");
  return res.data;
}

export async function clearROI(): Promise<{ status: string; message: string }> {
  const res = await api.post("/api/roi/clear");
  return res.data;
}

export async function detectROIViolations(
  vehicles: Array<{
    bbox: number[];
    track_id: number;
    class_name: string;
    conf: number;
  }>,
  redLightActive: boolean = false
): Promise<{ violations: any[] }> {
  const res = await api.post("/api/roi/detect", {
    vehicles,
    red_light_active: redLightActive,
  });
  return res.data;
}

// ======================== BEV (BIRD'S EYE VIEW) API ========================

export async function setBEVConfig(srcPoints: { x: number; y: number }[], dstPoints?: { x: number; y: number }[], stopLineY?: number): Promise<any> {
  const body: any = { src_points: srcPoints };
  if (dstPoints) body.dst_points = dstPoints;
  if (stopLineY !== undefined) body.stop_line_3d_y = stopLineY;
  try {
    const res = await api.post("/api/bev/config", body);
    return res.data;
  } catch {
    // Backend unavailable - silently fail
    return null;
  }
}

export async function getBEVConfig(): Promise<any> {
  const res = await api.get("/api/bev/config");
  return res.data;
}

export async function resetBEV(): Promise<any> {
  const res = await api.post("/api/bev/reset");
  return res.data;
}

// ======================== REPORTS API ========================

export interface OCRModelStats {
  model_name: string;
  dataset: {
    name: string;
    source: string;
    total_images: number;
    sample_data: Array<{
      name: string;
      label: string;
      type: number;
    }>;
  };
  evaluation_metrics: {
    plate_accuracy: number;
    character_accuracy: number;
    character_error_rate: number;
    average_latency_ms: number;
    fps: number;
  };
  evaluation_date: string;
  notes: string[];
}

export interface SpeedViolationReport {
  period_days: number;
  total_violations: number;
  speed_stats: {
    average: number;
    maximum: number;
    minimum: number;
  };
  by_vehicle_type: Array<{
    type: string;
    count: number;
    avg_speed: number;
  }>;
  by_hour: Array<{
    hour: number;
    count: number;
  }>;
  recent_violations: Array<{
    id: number;
    license_plate: string;
    vehicle_type: string;
    color: string;
    speed_kmh: number;
    speed_limit: number;
    timestamp: string;
    over_limit: number;
  }>;
}

export interface ComprehensiveReport {
  report_generated_at: string;
  system_overview: {
    total_vehicles_detected: number;
    total_violations: number;
    total_speed_violations: number;
    fraud_alerts: number;
  };
  violation_breakdown: Array<{
    type: string;
    count: number;
  }>;
  vehicle_distribution: Array<{
    type: string;
    color: string;
    count: number;
  }>;
  ocr_model_performance: OCRModelStats;
  speed_violation_analysis: SpeedViolationReport;
}

export async function getOCRReport(): Promise<OCRModelStats> {
  const res = await api.get("/api/reports/ocr");
  return res.data;
}

export async function getSpeedViolationsReport(days: number = 30): Promise<SpeedViolationReport> {
  const res = await api.get("/api/reports/speed-violations", { params: { days } });
  return res.data;
}

export async function getComprehensiveReport(): Promise<ComprehensiveReport> {
  const res = await api.get("/api/reports/comprehensive");
  return res.data;
}

export async function exportSystemReport(): Promise<{ status: string; filepath: string }> {
  const res = await api.post("/api/reports/export");
  return res.data;
}

export default api;
