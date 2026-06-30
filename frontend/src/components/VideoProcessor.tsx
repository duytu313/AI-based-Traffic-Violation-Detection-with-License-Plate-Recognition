"use client";

import { useState, useRef, useEffect } from "react";
import { Upload, X, Video, Play, Square, Car, MousePointer2 } from "lucide-react";
import { startVideoStream, stopVideoStream, getVideoFrame, getVideoStatus, getVideoDetected, setBEVConfig } from "@/lib/api";
import type { Config, ROIPoint } from "@/lib/types";
import ConfigPanel from "./ConfigPanel";
import BEVEditor from "./BEVEditor";

const defaultConfig: Config = {
  enable_violation_detection: true,
  enable_red_light_detection: false,
  enable_bev_detection: true,
  violation_conf_limit: 0.15,
  conf_more_than_two: 0.50,
  conf_no_helmet: 0.15,
  conf_using_mobile: 0.15,
  traffic_light_conf: 0.25,
  show_zones: false,
  show_bev: true,
  camera_direction: "down",
};

export default function VideoProcessor() {
  const [file, setFile] = useState<File | null>(null);
  const [config, setConfig] = useState<Config>(defaultConfig);
  const [loading, setLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [detected, setDetected] = useState<any>(null);
  const [showBEVEditor, setShowBEVEditor] = useState(false);
  const [bevPoints, setBEVPoints] = useState<ROIPoint[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const detectedIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const handleStartStream = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
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
      setStatus(null);
      setFrameUrl(null);
      setDetected(null);
      if (intervalRef.current) clearInterval(intervalRef.current);
    } catch {}
  };

  const fetchFrame = async () => {
    if (!isStreaming) return;
    try {
      const frameRes = await getVideoFrame();
      if (frameRes) {
        const blob = new Blob([frameRes], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        setFrameUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
      }
      const statusRes = await getVideoStatus();
      if (statusRes) setStatus(statusRes);
      const detectedRes = await getVideoDetected();
      if (detectedRes) setDetected(detectedRes);
    } catch {}
  };

  useEffect(() => {
    if (isStreaming) {
      fetchFrame();
      intervalRef.current = setInterval(fetchFrame, 50);
      detectedIntervalRef.current = setInterval(async () => {
        try {
          const detectedRes = await getVideoDetected();
          if (detectedRes) setDetected(detectedRes);
        } catch {}
      }, 250);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (detectedIntervalRef.current) clearInterval(detectedIntervalRef.current);
    };
  }, [isStreaming, fetchFrame]);

  const reset = () => {
    setFile(null);
    setFrameUrl(null);
    setStatus(null);
    setDetected(null);
    setError(null);
    setIsStreaming(false);
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
          <div className="card">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Video className="w-4 h-4" />
              Video Processing
            </h3>

            {/* Upload */}
            {!isStreaming && (
              <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/avi,video/mov,video/mkv"
                  onChange={handleFileChange}
                  className="hidden"
                  id="video-input"
                />
                <label htmlFor="video-input" className="cursor-pointer flex flex-col items-center gap-3">
                  <div className="w-16 h-16 rounded-full bg-purple-500/10 flex items-center justify-center">
                    <Upload className="w-8 h-8 text-purple-400" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-white">Select video to process</p>
                    <p className="text-sm text-slate-400 mt-1">MP4, AVI, MOV, MKV</p>
                  </div>
                </label>
              </div>
            )}

            {/* File info & controls */}
            {file && (
              <div className="mt-4 space-y-4">
                <div className="p-3 bg-slate-800/50 rounded-lg flex items-center justify-between">
                  <span className="text-sm text-slate-300 truncate">{file.name}</span>
                  {!isStreaming && (
                    <button onClick={reset} className="p-1 hover:bg-slate-700 rounded">
                      <X className="w-4 h-4 text-slate-400" />
                    </button>
                  )}
                </div>

                {!isStreaming && (
                  <button
                    onClick={handleStartStream}
                    disabled={loading}
                    className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                  >
                    <Play className="w-4 h-4" />
                    {loading ? "Processing..." : "Start video processing"}
                  </button>
                )}

                {isStreaming && (
                  <button
                    onClick={handleStopStream}
                    className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                  >
                    <Square className="w-4 h-4" />
                    Stop and select another video
                  </button>
                )}
              </div>
            )}

            {error && (
              <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-300">
                {error}
              </div>
            )}

            {/* Status */}
            {status && isStreaming && (
              <div className="mt-4 p-3 bg-slate-800/50 rounded-lg flex items-center gap-3 text-sm">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-slate-300">
                  Status: <strong>Running</strong>
                </span>
                <span className="text-slate-400">|</span>
                <span className="text-slate-300">
                  FPS: <strong className="text-yellow-400">{status.fps || 0}</strong>
                </span>
              </div>
            )}
          </div>

          {/* Video Stream Display */}
          {frameUrl && isStreaming && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-white flex items-center gap-2">
                  <Video className="w-4 h-4" />
                  Live Stream
                </h3>
                <button
                  onClick={() => setShowBEVEditor(!showBEVEditor)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                    showBEVEditor
                      ? "bg-cyan-600 hover:bg-cyan-700 text-white"
                      : "bg-slate-700 hover:bg-slate-600 text-white"
                  }`}
                >
                  🛠️ {showBEVEditor ? "Drawing BEV..." : "Draw 3D Zone"}
                </button>
              </div>

              {/* BEV Editor */}
              {showBEVEditor && (
                <div className="mb-4">
                  <BEVEditor
                    imageUrl={frameUrl}
                    initialPoints={bevPoints}
                    onSave={async (points) => {
                      setBEVPoints(points);
                      try {
                        await setBEVConfig(points);
                      } catch {
                        // Backend unavailable - points are saved locally
                      }
                      setShowBEVEditor(false);
                    }}
                    onCancel={() => setShowBEVEditor(false)}
                  />
                </div>
              )}

              {/* Video Frame */}
              {!showBEVEditor && (
                <img
                  src={frameUrl}
                  alt="Video stream"
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
              Recent Detections
            </h3>

            {!detected || (detected.vehicles.length === 0 && detected.violations.length === 0) ? (
              <p className="text-sm text-slate-400 text-center py-8">
                No vehicles detected
              </p>
            ) : (
              <div className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
                {/* Vehicles */}
                {detected.vehicles.map((v: any, i: number) => (
                  <div key={i} className="p-2 bg-slate-800/50 rounded-lg space-y-2">
                    {/* Vehicle crop */}
                    {v.vehicle_b64 && (
                      <div>
                        <p className="text-xs text-slate-400 mb-1">Vehicle</p>
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
                        <p className="text-xs text-slate-400 mb-1">License Plate</p>
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
                        <p className="text-slate-300">License Plate: <span className="text-white">{v.plate_text}</span></p>
                      )}
                    </div>
                  </div>
                ))}

                {/* Violations */}
                {detected.violations.map((v: any, i: number) => (
                  <div key={i} className="p-2 bg-red-500/10 border border-red-500/20 rounded-lg space-y-2">
                    {v.crop_b64 && (
                      <div>
                        <p className="text-xs text-red-400 mb-1">Violation</p>
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
                        <p className="text-slate-400">License Plate: {v.plate_text}</p>
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