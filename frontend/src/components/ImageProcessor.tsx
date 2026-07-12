"use client";

import { useState, useCallback, useRef } from "react";
import { Upload, Image as ImageIcon, AlertTriangle, CheckCircle2, X } from "lucide-react";
import { processImage } from "@/lib/api";
import type { ProcessImageResponse, Config } from "@/lib/types";
import ConfigPanel from "./ConfigPanel";

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

export default function ImageProcessor() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [config, setConfig] = useState<Config>(defaultConfig);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ProcessImageResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragActive, setIsDragActive] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const f = acceptedFiles[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
      setError(null);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onDrop(Array.from(files));
    }
  }, [onDrop]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
    const files = Array.from(e.dataTransfer.files);
    const imageFiles = files.filter(f => f.type.startsWith('image/'));
    if (imageFiles.length > 0) {
      onDrop(imageFiles);
    }
  }, [onDrop]);

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  const handleProcess = async () => {
    if (!file || loading) return;  // Prevent double-click
    setLoading(true);
    setError(null);
    try {
      const res = await processImage(file, config);
      setResult(res);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "Image processing error");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const getViolationIcon = (type: string) => {
    if (type.includes("MOBILE")) return "📱";
    if (type.includes("MORE_THAN_TWO")) return "🛵";
    if (type.includes("RED_LIGHT")) return "🚨";
    return "⛑️";
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Left sidebar - Config */}
      <div className="lg:col-span-1 space-y-4">
        <ConfigPanel config={config} onChange={setConfig} />
      </div>

      {/* Main content */}
      <div className="lg:col-span-3 space-y-6">
        {/* Dropzone */}
        {!preview && (
          <div
            onClick={openFileDialog}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`dropzone ${isDragActive ? "active" : ""}`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              onChange={handleFileSelect}
              className="hidden"
            />
            <div className="flex flex-col items-center gap-3">
              <div className="w-16 h-16 rounded-full bg-blue-500/10 flex items-center justify-center">
                <Upload className="w-8 h-8 text-blue-400" />
              </div>
              <div>
                <p className="text-lg font-medium text-white">
                  Drag and drop image here
                </p>
                <p className="text-sm text-slate-400 mt-1">
                  or click to select file (JPG, PNG)
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Preview */}
        {preview && (
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-white flex items-center gap-2">
                <ImageIcon className="w-4 h-4" />
                Input Image
              </h3>
              <button
                onClick={reset}
                className="p-1.5 rounded-lg hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <img
              src={preview}
              alt="Preview"
              className="w-full max-h-96 object-contain rounded-lg"
            />
            <div className="flex gap-3 mt-4">
              <button
                onClick={handleProcess}
                disabled={loading}
                className="flex-1 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:text-slate-400 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="spinner w-4 h-4" />
                    Processing...
                  </>
                ) : (
                  <>
                    <ImageIcon className="w-4 h-4" />
                    Process Image
                  </>
                )}
              </button>
              <button
                onClick={reset}
                disabled={loading}
                className="px-4 py-2.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
              >
                Choose Another Image
              </button>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <p className="text-red-300 text-sm">{error}</p>
          </div>
        )}

        {/* Loading overlay */}
        {loading && (
          <div className="processing-overlay">
            <div className="card text-center space-y-4">
              <div className="spinner mx-auto" />
              <p className="text-white font-medium">Processing image...</p>
              <p className="text-sm text-slate-400">
                System is detecting vehicles, license plates, and violations
              </p>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="card text-center">
                <p className="text-2xl font-bold text-blue-400">
                  {result.stats.total_vehicles}
                </p>
                <p className="text-xs text-slate-400 mt-1">Vehicles</p>
              </div>
              <div className="card text-center">
                <p className="text-2xl font-bold text-red-400">
                  {result.stats.total_violations}
                </p>
                <p className="text-xs text-slate-400 mt-1">Violations</p>
              </div>
              <div className="card text-center">
                <p className="text-2xl font-bold text-yellow-400">
                  {result.stats.total_red_light}
                </p>
                <p className="text-xs text-slate-400 mt-1">Red Light Violations</p>
              </div>
            </div>

            {/* Result Image */}
            <div className="card">
              <h3 className="font-semibold text-white mb-4">Processing Results</h3>
              <img
                src={`data:image/jpeg;base64,${result.image_base64}`}
                alt="Result"
                className="w-full rounded-lg"
              />
            </div>

            {/* Violations Detail */}
            {result.violations.length > 0 && (
              <div className="card">
                <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                  Violation Details
                </h3>
                <div className="space-y-4">
                  {result.violations.map((v, i) => (
                    <div
                      key={i}
                      className="p-3 bg-red-500/5 border border-red-500/10 rounded-lg"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-white">
                          {v.color} {v.vtype}
                        </span>
                        {v.plate_text && (
                          <span className="badge bg-blue-500/10 text-blue-400">
                            {v.plate_text}
                          </span>
                        )}
                      </div>
                      <div className="space-y-1">
                        {v.violations.map((viol, j) => (
                          <div
                            key={j}
                            className="flex items-center gap-2 text-sm text-red-300"
                          >
                            <span>{getViolationIcon(viol.type)}</span>
                            <span>{viol.details}</span>
                            <span className="text-slate-500">
                              ({(viol.conf * 100).toFixed(1)}%)
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Matched Plates */}
            {result.matched_plates.length > 0 && (
              <div className="card">
                <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400" />
                  Recognized License Plates
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {result.matched_plates.map((p, i) => (
                    <div
                      key={i}
                      className="p-3 bg-slate-800/50 rounded-lg flex items-center justify-between"
                    >
                      <div>
                        <p className="text-sm text-slate-300">
                          {p.color} {p.vtype}
                        </p>
                      </div>
                      <span className="badge bg-green-500/10 text-green-400 text-sm">
                        {p.plate_text}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* No violations */}
            {result.violations.length === 0 && result.matched_plates.length > 0 && (
              <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg flex items-center gap-3">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                <p className="text-green-300 text-sm">
                   ✅ No violations detected. All vehicles are complying with traffic laws.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}