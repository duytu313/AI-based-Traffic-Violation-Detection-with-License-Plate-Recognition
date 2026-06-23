"use client";

import { useState, useRef, useEffect } from "react";
import { Camera, StopCircle, AlertTriangle, Car, User, MousePointer2 } from "lucide-react";
import { startWebcam, stopWebcam, getWebcamFrame, getWebcamStatus, getWebcamDetected, setROI, getROI, clearROI } from "@/lib/api";
import type { Config, ROIPoint } from "@/lib/types";
import ConfigPanel from "./ConfigPanel";
import ROIEditor from "./ROIEditor";

const defaultConfig: Config = {
  enable_violation_detection: true,
  enable_red_light_detection: false,
  violation_conf_limit: 0.15,
  conf_more_than_two: 0.50,
  conf_no_helmet: 0.15,
  conf_using_mobile: 0.15,
  traffic_light_conf: 0.25,
  show_zones: false,
  camera_direction: "down",
};

export default function WebcamProcessor() {
  const [config, setConfig] = useState<Config>(defaultConfig);
  const [loading, setLoading] = useState(false);
  const [webcamSource, setWebcamSource] = useState("0");
  const [webcamBackend, setWebcamBackend] = useState("AUTO");
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<any>(null);
  const [detected, setDetected] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [showROIEditor, setShowROIEditor] = useState(false);
  const [roiPoints, setROIPoints] = useState<ROIPoint[]>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const startWebcamHandler = async () => {
    setLoading(true);
    setError(null);
    try {
      await startWebcam(webcamSource, webcamBackend);
      setIsStreaming(true);
      setStatus({ running: true, source: webcamSource });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const stopWebcamHandler = async () => {
    try {
      await stopWebcam();
      setIsStreaming(false);
      setStatus(null);
      setDetected(null);
      setFrameUrl(null);
      if (intervalRef.current) clearInterval(intervalRef.current);
    } catch {}
  };

  const fetchFrame = async () => {
    if (!isStreaming) return;
    try {
      const frameRes = await getWebcamFrame();
      if (frameRes) {
        const blob = new Blob([frameRes], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        setFrameUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
      }
      const statusRes = await getWebcamStatus();
      if (statusRes) setStatus(statusRes);
      const detectedRes = await getWebcamDetected();
      if (detectedRes) setDetected(detectedRes);
    } catch {}
  };

  // Load existing ROI on mount
  useEffect(() => {
    const loadROI = async () => {
      try {
        const roiRes = await getROI();
        if (roiRes.roi_config && roiRes.roi_config.points.length > 0) {
          setROIPoints(roiRes.roi_config.points);
        }
      } catch {}
    };
    loadROI();
  }, []);

  useEffect(() => {
    if (isStreaming) {
      fetchFrame();
      intervalRef.current = setInterval(fetchFrame, 50);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isStreaming, fetchFrame]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Config Panel */}
        <div className="lg:col-span-1">
          <ConfigPanel config={config} onChange={setConfig} />
        </div>

        {/* Main Content - Video Stream */}
        <div className="lg:col-span-2 space-y-6">
          <div className="card">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Camera className="w-4 h-4" />
              Webcam / RTSP Stream
            </h3>

            {/* Controls */}
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Nguồn (index/URL)</label>
                  <input
                    type="text"
                    value={webcamSource}
                    onChange={(e) => setWebcamSource(e.target.value)}
                    placeholder="0 hoặc http://..."
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Backend</label>
                  <select
                    value={webcamBackend}
                    onChange={(e) => setWebcamBackend(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="AUTO">AUTO</option>
                    <option value="FFMPEG">FFMPEG</option>
                    <option value="GSTREAMER">GSTREAMER</option>
                  </select>
                </div>
                <div className="flex items-end gap-2">
                  {!isStreaming ? (
                    <button
                      onClick={startWebcamHandler}
                      disabled={loading}
                      className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                    >
                      <Camera className="w-4 h-4" />
                      {loading ? "Đang mở..." : "Bắt đầu"}
                    </button>
                  ) : (
                    <button
                      onClick={stopWebcamHandler}
                      className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                    >
                      <StopCircle className="w-4 h-4" />
                      Dừng
                    </button>
                  )}
                </div>
              </div>

              {status && (
                <div className="p-3 bg-slate-800/50 rounded-lg flex items-center gap-3 text-sm">
                  <span className={`w-2 h-2 rounded-full ${status.running ? "bg-green-400 animate-pulse" : "bg-red-400"}`} />
                  <span className="text-slate-300">
                    Trạng thái: <strong>{status.running ? "Đang chạy" : "Dừng"}</strong>
                  </span>
                  <span className="text-slate-400">|</span>
                  <span className="text-slate-300">
                    FPS: <strong className="text-yellow-400">{status.fps || 0}</strong>
                  </span>
                </div>
              )}

              {error && (
                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-300">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* Video Stream Display */}
          {frameUrl && isStreaming && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-white">Luồng trực tiếp</h3>
                <button
                  onClick={() => setShowROIEditor(!showROIEditor)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                    showROIEditor
                      ? "bg-yellow-600 hover:bg-yellow-700 text-white"
                      : "bg-slate-700 hover:bg-slate-600 text-white"
                  }`}
                >
                  <MousePointer2 className="w-4 h-4" />
                  {showROIEditor ? "Đang vẽ ROI..." : "Vẽ vùng vi phạm"}
                </button>
              </div>

              {/* ROI Editor */}
              {showROIEditor && (
                <div className="mb-4">
                  <ROIEditor
                    imageUrl={frameUrl}
                    initialPoints={roiPoints}
                    onSave={(points) => {
                      setROIPoints(points);
                      setROI(points, "violation_zone");
                      setShowROIEditor(false);
                    }}
                    onCancel={() => setShowROIEditor(false)}
                  />
                </div>
              )}

              {/* Webcam Frame */}
              {!showROIEditor && (
                <img
                  src={frameUrl}
                  alt="Webcam"
                  className="w-full rounded-lg"
                  style={{ maxHeight: "600px", objectFit: "contain" }}
                />
              )}
            </div>
          )}
        </div>

        {/* Right Sidebar - Detected Items */}
        <div className="lg:col-span-1">
          <div className="card sticky top-20">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Car className="w-4 h-4" />
              Phát hiện gần đây
            </h3>

            {!detected || (detected.vehicles.length === 0 && detected.violations.length === 0) ? (
              <p className="text-sm text-slate-400 text-center py-8">
                Chưa phát hiện xe nào
              </p>
            ) : (
              <div className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
                {/* Vehicles */}
                {detected.vehicles.map((v: any, i: number) => (
                  <div key={i} className="p-2 bg-slate-800/50 rounded-lg space-y-2">
                    {/* Vehicle crop */}
                    {v.vehicle_b64 && (
                      <div>
                        <p className="text-xs text-slate-400 mb-1">Xe</p>
                        <img
                          src={`data:image/jpeg;base64,${v.vehicle_b64}`}
                          alt="Vehicle"
                          className="w-full rounded"
                        />
                      </div>
                    )}

                    {/* Plate crop */}
                    {v.plate_b64 && (
                      <div>
                        <p className="text-xs text-slate-400 mb-1">Biển số</p>
                        <img
                          src={`data:image/jpeg;base64,${v.plate_b64}`}
                          alt="Plate"
                          className="w-full rounded border border-green-500/50"
                        />
                      </div>
                    )}

                    {/* Info */}
                    <div className="text-xs space-y-1">
                      <p className="text-green-400 font-medium">{v.info}</p>
                      {v.plate_text && (
                        <p className="text-slate-300">Biển số: <span className="text-white">{v.plate_text}</span></p>
                      )}
                    </div>
                  </div>
                ))}

                {/* Violations */}
                {detected.violations.map((v: any, i: number) => (
                  <div key={i} className="p-2 bg-red-500/10 border border-red-500/20 rounded-lg space-y-2">
                    {v.crop_b64 && (
                      <div>
                        <p className="text-xs text-red-400 mb-1">Vi phạm</p>
                        <img
                          src={`data:image/jpeg;base64,${v.crop_b64}`}
                          alt="Violation"
                          className="w-full rounded"
                        />
                      </div>
                    )}
                    <div className="text-xs space-y-1">
                      <p className="text-red-300 font-medium">{v.details}</p>
                      {v.plate_text && (
                        <p className="text-slate-400">Biển số: {v.plate_text}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}