"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Upload, X, Video, Play, Square, Car, MousePointer2 } from "lucide-react";
import { startVideoStream, stopVideoStream, getVideoFrame, getVideoStatus, getVideoDetected, setBEVConfig, pauseVideoStream, resumeVideoStream, healthCheck } from "@/lib/api";
import type { Config, ROIPoint } from "@/lib/types";
import ConfigPanel from "./ConfigPanel";
import BEVEditor from "./BEVEditor";

const defaultConfig: Config = {
  enable_violation_detection: false,
  enable_red_light_detection: false,
  enable_bev_detection: false,
  enable_speed_violation_detection: false,
  violation_conf_limit: 0.15,
  conf_more_than_two: 0.50,
  conf_no_helmet: 0.15,
  conf_using_mobile: 0.15,
  traffic_light_conf: 0.25,
  speed_limit: 60,
  show_zones: false,
  show_bev: false,
  camera_direction: "down",
};

export default function VideoProcessor() {
  const [file, setFile] = useState<File | null>(null);
  const [config, setConfig] = useState<Config>(defaultConfig);
  const [loading, setLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [status, setStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [detected, setDetected] = useState<any>(null);
  const [showBEVEditor, setShowBEVEditor] = useState(false);
  const [bevPoints, setBEVPoints] = useState<ROIPoint[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const detectedIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const isPausedRef = useRef(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const handleStartStream = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setDetected(null);
    setFrameUrl(null);
    setIsPaused(false);
    isPausedRef.current = false;
    try {
      await startVideoStream(file, config);
      setIsStreaming(true);
      setStatus({ running: true });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStopStream = async () => {
    try {
      await stopVideoStream();
      setIsStreaming(false);
      setIsPaused(false);
      isPausedRef.current = false;
      setStatus(null);
      setFrameUrl(null);
      setDetected(null);
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (detectedIntervalRef.current) clearInterval(detectedIntervalRef.current);
    } catch {}
  };

  const handleDrawZone = async () => {
    const willShow = !showBEVEditor;
    setShowBEVEditor(willShow);
    
    if (willShow && isStreaming) {
      // Pause the video when opening BEV editor
      try {
        await pauseVideoStream();
        setIsPaused(true);
        isPausedRef.current = true;
      } catch {}
    } else if (!willShow && isPausedRef.current) {
      // Resume when cancelling/closing
      try {
        await resumeVideoStream();
        setIsPaused(false);
        isPausedRef.current = false;
      } catch {}
    }
  };

  const handleBEVSave = async (points: ROIPoint[]) => {
    setBEVPoints(points);
    try {
      await setBEVConfig(points);
    } catch {
      // Backend unavailable - points are saved locally
    }
    setShowBEVEditor(false);
    
    // Resume video after save
    if (isPausedRef.current) {
      try {
        await resumeVideoStream();
      } catch {}
      setIsPaused(false);
      isPausedRef.current = false;
    }
  };

  const handleBEVCancel = () => {
    setShowBEVEditor(false);
    
    // Resume video after cancel
    if (isPausedRef.current) {
      resumeVideoStream().catch(() => {});
      setIsPaused(false);
      isPausedRef.current = false;
    }
  };

  const fetchFrame = useCallback(async () => {
    if (!isStreaming) return;
    
    // Check backend health first
    try {
      const healthy = await healthCheck();
      if (!healthy) {
        console.error("Backend is not running");
        setError("Backend server is not running. Please start the backend server (python backend/main.py)");
        return;
      }
    } catch (err) {
      console.error("Health check failed:", err);
      // Don't stop streaming on health check failure, might be temporary
    }
    
    // Fetch frame, status, and detected independently so one failure doesn't block others
    try {
      const frameRes = await getVideoFrame();
      if (frameRes && frameRes.size > 0) {
        const blob = new Blob([frameRes], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        setFrameUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
      } else {
        // Empty response means video ended - auto stop
        console.log("Video ended, auto-stopping...");
        handleStopStream();
        return;
      }
    } catch (err: any) {
      console.error("Fetch frame error:", err);
      // Don't stop on network errors - might be temporary
      if (err?.message?.includes("Network Error")) {
        setError("Cannot connect to backend. Make sure backend is running on http://localhost:8000");
      }
    }
    
    try {
      const statusRes = await getVideoStatus();
      if (statusRes) {
        setStatus(statusRes);
        // If backend says not running, auto-stop
        if (!statusRes.running && isStreaming) {
          console.log("Video processing ended");
          setIsStreaming(false);
          if (intervalRef.current) clearInterval(intervalRef.current);
          if (detectedIntervalRef.current) clearInterval(detectedIntervalRef.current);
          return;
        }
      }
    } catch (err: any) {
      console.error("Fetch status error:", err);
    }
    
    try {
      const detectedRes = await getVideoDetected();
      if (detectedRes) setDetected(detectedRes);
    } catch (err: any) {
      console.error("Fetch detected error:", err);
    }
  }, [isStreaming]);

  useEffect(() => {
    if (isStreaming) {
      fetchFrame();
      intervalRef.current = setInterval(fetchFrame, 100);
      detectedIntervalRef.current = setInterval(async () => {
        try {
          const detectedRes = await getVideoDetected();
          if (detectedRes) setDetected(detectedRes);
        } catch (err: any) {
          console.error("Detected polling error:", err);
        }
      }, 500);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (detectedIntervalRef.current) clearInterval(detectedIntervalRef.current);
    };
  }, [isStreaming, fetchFrame]);

  // Health check and auto-reconnect
  useEffect(() => {
    if (!isStreaming) return;
    
    const healthCheckInterval = setInterval(async () => {
      try {
        const isHealthy = await healthCheck();
        if (!isHealthy) {
          console.warn("Backend health check failed");
          setError("Backend connection unstable. Please check if backend is running.");
        } else if (error) {
          // Clear error if backend is healthy again
          setError(null);
        }
      } catch (err: any) {
        console.error("Health check error:", err);
        setError(`Connection lost: ${err.message}`);
      }
    }, 5000); // Check every 5 seconds
    
    return () => clearInterval(healthCheckInterval);
  }, [isStreaming, error]);

  const reset = () => {
    setFile(null);
    setFrameUrl(null);
    setStatus(null);
    setDetected(null);
    setError(null);
    setIsStreaming(false);
    setIsPaused(false);
    isPausedRef.current = false;
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Config Panel */}
        <div className="lg:col-span-1">
          <ConfigPanel config={config} onChange={setConfig} />
        </div>

        {/* Main Content - Video Stream */}
        <div className="lg:col-span-2 space-y-6">
          {/* Live Stream Section */}
          <div className="card">
            {/* Header with controls */}
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-white flex items-center gap-2">
                <Video className="w-4 h-4" />
                Live Stream
              </h3>
              <div className="flex items-center gap-2">
                {/* Start/Stop Button */}
                {file && (
                  <button
                    onClick={isStreaming ? handleStopStream : handleStartStream}
                    disabled={loading}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                      isStreaming
                        ? "bg-red-600 hover:bg-red-700 text-white"
                        : "bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 text-white"
                    }`}
                  >
                    {isStreaming ? (
                      <>
                        <Square className="w-4 h-4" />
                        Stop
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        {loading ? "Processing..." : "Start"}
                      </>
                    )}
                  </button>
                )}
                
                {/* Draw 3D Zone Button - only when streaming */}
                {isStreaming && (
                  <button
                    onClick={handleDrawZone}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                      showBEVEditor
                        ? "bg-cyan-600 hover:bg-cyan-700 text-white"
                        : "bg-slate-700 hover:bg-slate-600 text-white"
                    }`}
                  >
                    🛠️ {showBEVEditor ? "Drawing BEV..." : "Draw 3D Zone"}
                  </button>
                )}

                {/* Pause indicator */}
                {isPaused && (
                  <span className="text-xs text-yellow-400 font-medium">⏸ PAUSED</span>
                )}
              </div>
            </div>

            {/* BEV Editor */}
            {showBEVEditor && (
              <div className="mb-4">
                <BEVEditor
                  imageUrl={frameUrl || ""}
                  initialPoints={bevPoints}
                  onSave={handleBEVSave}
                  onCancel={handleBEVCancel}
                />
              </div>
            )}

            {/* Video Frame - Always visible */}
            <div className="relative bg-slate-900/50 rounded-lg overflow-hidden mb-3" style={{ minHeight: "400px" }}>
              {frameUrl ? (
                <img
                  src={frameUrl}
                  alt="Video stream"
                  className="w-full rounded-lg"
                  style={{ maxHeight: "600px", objectFit: "contain" }}
                />
              ) : (
                <div className="flex items-center justify-center" style={{ minHeight: "400px" }}>
                  <div className="text-center">
                    <Video className="w-16 h-16 text-slate-600 mx-auto mb-3" />
                    <p className="text-slate-400">Start video to see live stream</p>
                  </div>
                </div>
              )}
              {/* Paused overlay */}
              {isPaused && (
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center rounded-lg">
                  <p className="text-yellow-400 font-bold text-lg">⏸ PAUSED - Drawing 3D Zone</p>
                </div>
              )}
              {/* Video ended overlay */}
              {!isStreaming && status && !status.running && frameUrl && (
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center rounded-lg">
                  <p className="text-white font-bold text-lg">Video ended - Press Start to replay</p>
                </div>
              )}
            </div>

            {/* Upload - Compact - Integrated with Live Stream */}
            {!file && !isStreaming && (
              <div className="border-2 border-dashed border-slate-600 rounded-lg p-2 text-center cursor-pointer" style={{ minHeight: "40px" }}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/avi,video/mov,video/mkv"
                  onChange={handleFileChange}
                  className="hidden"
                  id="video-input"
                />
                <label htmlFor="video-input" className="cursor-pointer flex flex-col items-center gap-1">
                  <Upload className="w-5 h-5 text-purple-400" />
                  <div>
                    <p className="text-sm font-medium text-white">Select video</p>
                    <p className="text-xs text-slate-400">MP4, AVI, MOV, MKV</p>
                  </div>
                </label>
              </div>
            )}

            {/* File info - shown when file is selected */}
            {file && !isStreaming && (
              <div className="p-2 bg-slate-800/50 rounded-lg flex items-center justify-between">
                <span className="text-sm text-slate-300 truncate">{file.name}</span>
                <button onClick={reset} className="p-1 hover:bg-slate-700 rounded">
                  <X className="w-4 h-4 text-slate-400" />
                </button>
              </div>
            )}

            {error && (
              <div className="mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-300">
                {error}
              </div>
            )}
          </div>

          {/* Detected Violations Section - Horizontal 3 Columns - Always Visible */}
          <div className="card">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Car className="w-4 h-4" />
              Detected Violations
            </h3>

            <div className="grid grid-cols-3 gap-3">
              {[0, 1, 2].map((index) => {
                const violation = detected?.violations?.[index];
                const hasViolation = violation?.details;
                
                return (
                  <div
                    key={index}
                    className={`p-3 rounded-lg border-2 transition-all ${
                      hasViolation
                        ? "bg-red-500/10 border-red-500/20"
                        : "bg-slate-800/20 border-slate-700 border-dashed"
                    }`}
                    style={{ minHeight: "180px" }}
                  >
                    {hasViolation ? (
                      <>
                        {/* License Plate - Big & Bold at the top */}
                        <div className="text-center mb-2">
                          {violation.plate_text && violation.plate_text.trim() !== "" ? (
                            <p className="text-lg font-bold text-green-400 font-mono tracking-wider bg-green-500/10 rounded-lg py-1 px-2 inline-block">
                              {violation.plate_text}
                            </p>
                          ) : (
                            <p className="text-sm text-slate-500 italic">No plate detected</p>
                          )}
                        </div>

                        {/* Vehicle/Violation Image */}
                        {violation.crop_b64 && (
                          <div className="mb-2 flex items-center justify-center bg-slate-900/50 rounded-lg p-2" style={{ minHeight: "100px" }}>
                            <img
                              src={`data:image/jpeg;base64,${violation.crop_b64}`}
                              alt="Violation"
                              className="w-full rounded"
                              style={{ maxHeight: "100px", objectFit: "contain" }}
                            />
                          </div>
                        )}

                        {/* Violation Details */}
                        <div className="space-y-1">
                          <p className="text-sm font-bold text-red-300 text-center">
                            {violation.details}
                          </p>
                          
                          {/* Vehicle Info - Always show when violation exists */}
                          <div className="text-xs space-y-0.5 mt-2">
                            {/* Vehicle Type */}
                            {(violation.vehicle_type || violation.info) && (
                              <p className="text-slate-300">
                                <span className="text-slate-400">Type:</span> <span className="text-white">{violation.vehicle_type || violation.info}</span>
                              </p>
                            )}
                            {/* Color */}
                            {violation.color && (
                              <p className="text-slate-300">
                                <span className="text-slate-400">Color:</span> 
                                <span className="inline-block w-3 h-3 rounded-full ml-1 align-middle" style={{ backgroundColor: violation.color.toLowerCase() }}></span>
                                <span className="text-white ml-1">{violation.color}</span>
                              </p>
                            )}
                            {/* Fallback: show info if no vehicle_type/color */}
                            {!violation.vehicle_type && !violation.color && violation.info && (
                              <p className="text-slate-300">
                                <span className="text-slate-400">Info:</span> <span className="text-white">{violation.info}</span>
                              </p>
                            )}
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="flex items-center justify-center" style={{ minHeight: "160px" }}>
                        <div className="text-center">
                          <p className="text-xs text-slate-500">Slot {index + 1}</p>
                          <p className="text-xs text-slate-600 mt-1">Waiting for violation...</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {detected && detected.violations && detected.violations.length > 3 && (
              <p className="text-xs text-slate-400 text-center mt-3">
                Showing latest 3 of {detected.violations.length} violations
              </p>
            )}
          </div>
        </div>

        {/* Right Sidebar - Detected License Plates */}
        <div className="lg:col-span-1">
          <div className="card sticky top-20">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Car className="w-4 h-4" />
              Detected License Plates
            </h3>

            <div className="space-y-3">
              {/* 3 fixed slots for license plates */}
              {[0, 1, 2].map((index) => {
                const vehicle = detected?.vehicles?.[index];
                const hasPlate = vehicle?.plate_text;
                
                return (
                  <div
                    key={index}
                    className={`p-3 rounded-lg border-2 transition-all ${
                      hasPlate
                        ? "bg-slate-800/50 border-green-500/50"
                        : "bg-slate-800/20 border-slate-700 border-dashed"
                    }`}
                    style={{ minHeight: "160px" }}
                  >
                    {hasPlate ? (
                      <div className="flex gap-3 items-center h-full">
                        {/* License plate image - 2/3 width */}
                        {vehicle.plate_b64 && (
                          <div className="w-2/3 flex items-center justify-center">
                            <img
                              src={`data:image/jpeg;base64,${vehicle.plate_b64}`}
                              alt="License Plate"
                              className="w-full rounded"
                              style={{ maxHeight: "140px", objectFit: "contain" }}
                            />
                          </div>
                        )}
                        {/* License plate text - 1/3 width */}
                        <div className={vehicle.plate_b64 ? "w-1/3" : "w-full flex items-center justify-center"}>
                          <p className="text-base font-bold text-white font-mono text-center whitespace-nowrap overflow-hidden text-ellipsis" title={vehicle.plate_text}>
                            {vehicle.plate_text}
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center" style={{ minHeight: "140px" }}>
                        <div className="text-center">
                          <p className="text-xs text-slate-500">Slot {index + 1}</p>
                          <p className="text-xs text-slate-600 mt-1">Waiting for detection...</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {!detected || detected.vehicles.length === 0 ? (
              <p className="text-xs text-slate-500 text-center mt-3">
                Start video to detect license plates
              </p>
            ) : detected.vehicles.length > 3 && (
              <p className="text-xs text-slate-400 text-center mt-3">
                Showing latest 3 of {detected.vehicles.length} detections
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}