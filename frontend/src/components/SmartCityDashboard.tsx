"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  Camera,
  Clock,
  Map as MapIcon,
  RefreshCw,
  Route,
  TrafficCone,
  XCircle,
} from "lucide-react";
import {
  getStats,
  getVehicles,
  getViolations,
  getSmartCityCameraDetected,
  getSmartCityCameraFrame,
  getSmartCityCameraStatus,
  startSmartCityCamera,
  stopSmartCityCamera,
} from "@/lib/api";
import type { DBVehicle, DBViolation, Stats } from "@/lib/types";
import type {
  SmartCityCamera,
} from "@/lib/api";

type CameraState = {
  source: string;
  running: boolean;
  fps: number;
  plate: string;
  info: string;
  frameUrl?: string;
  count: number;
};

function formatTime(value?: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleString("vi-VN");
}

function getHour(value: string) {
  return new Date(value).getHours();
}

function violationLabel(type: string) {
  const labels: Record<string, string> = {
    WITHOUT_HELMET: "Không đội mũ",
    MORE_THAN_TWO_PERSONS: "Chở quá người",
    USING_MOBILE: "Dùng điện thoại",
    RED_LIGHT_VIOLATION: "Vượt đèn đỏ",
  };
  return labels[type] || type;
}

const cameraLabels: Record<SmartCityCamera, { label: string; location: string }> = {
  cam1: { label: "CAM 1", location: "Ngã tư chính" },
  cam2: { label: "CAM 2", location: "Đường vành đai" },
  cam3: { label: "CAM 3", location: "Khu dân cư" },
  cam4: { label: "CAM 4", location: "Nút giao thông" },
};

function CameraMonitor({
  camera,
  state,
  onStart,
  onStop,
  onRefresh,
}: {
  camera: SmartCityCamera;
  state: CameraState;
  onStart: () => void;
  onStop: () => void;
  onRefresh: () => void;
}) {
  const meta = cameraLabels[camera];

  return (
    <section className="card space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/10">
            <Camera className="h-4 w-4 text-blue-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">{meta.label}</h3>
            <p className="text-xs text-slate-400">{meta.location}</p>
          </div>
        </div>
        <span className={`badge text-xs ${state.running ? "bg-emerald-500/10 text-emerald-400" : "bg-slate-700 text-slate-400"}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${state.running ? "bg-emerald-400" : "bg-slate-500"}`} />
          {state.running ? `${state.fps.toFixed(1)} FPS` : "Tắt"}
        </span>
      </div>

      <div className="relative aspect-video overflow-hidden rounded-lg border border-slate-700 bg-slate-950">
        {state.frameUrl ? (
          <div className="absolute inset-0 bg-cover bg-center" style={{ backgroundImage: `url(${state.frameUrl})` }} />
        ) : (
          <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(30,41,59,0.9),rgba(2,6,23,0.95))]" />
        )}
        <div className="absolute inset-0 bg-slate-950/25" />
        <div className="absolute left-3 top-3 rounded border border-white/20 px-2 py-0.5 text-[10px] font-medium text-white/80">
          {meta.label} • {meta.location}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-xs">
        <input
          value={state.source}
          onChange={() => {}}
          placeholder="Nguồn RTSP / webcam"
          className="col-span-2 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <div className="flex items-center justify-center rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-300">
          {state.count} xe
        </div>
      </div>

      {state.plate && (
        <div className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-300">
          <span className="text-slate-500">Biển số: </span>
          {state.plate}
        </div>
      )}

      <div className="flex gap-1.5">
        <button
          onClick={onStart}
          className="inline-flex flex-1 items-center justify-center gap-1 rounded bg-emerald-600 px-2 py-1.5 text-xs font-medium text-white transition-colors hover:bg-emerald-500"
        >
          <Camera className="h-3 w-3" />
          {state.running ? "Live" : "Bật"}
        </button>
        <button
          onClick={onRefresh}
          className="inline-flex items-center justify-center rounded border border-slate-700 px-2 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:bg-slate-800"
        >
          <RefreshCw className="h-3 w-3" />
        </button>
        <button
          onClick={onStop}
          className="inline-flex items-center justify-center rounded border border-red-700 px-2 py-1.5 text-xs font-medium text-red-300 transition-colors hover:bg-red-800"
        >
          <XCircle className="h-3 w-3" />
        </button>
      </div>
    </section>
  );
}

export default function SmartCityDashboard() {
  const [vehicles, setVehicles] = useState<DBVehicle[]>([]);
  const [violations, setViolations] = useState<DBViolation[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  // 4 cameras state
  const [cameras, setCameras] = useState<Record<SmartCityCamera, CameraState>>({
    cam1: { source: "0", running: false, fps: 0, plate: "", info: "", count: 0 },
    cam2: { source: "1", running: false, fps: 0, plate: "", info: "", count: 0 },
    cam3: { source: "2", running: false, fps: 0, plate: "", info: "", count: 0 },
    cam4: { source: "3", running: false, fps: 0, plate: "", info: "", count: 0 },
  });

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [vehicleRows, violationRows, statRows] = await Promise.all([
        getVehicles({ limit: 500, offset: 0 }),
        getViolations({ limit: 200, offset: 0 }),
        getStats(),
      ]);
      setVehicles(vehicleRows);
      setViolations(violationRows);
      setStats(statRows);
    } finally {
      setLoading(false);
    }
  }, []);

  // Refresh cameras
  const refreshCamera = useCallback(async (camera: SmartCityCamera) => {
    try {
      const [status, detected] = await Promise.all([
        getSmartCityCameraStatus(camera),
        getSmartCityCameraDetected(camera),
      ]);
      let frameUrl: string | undefined;
      // Only try to get frame if camera is running
      if (status.running) {
        try {
          const blob = await getSmartCityCameraFrame(camera);
          frameUrl = URL.createObjectURL(blob);
        } catch {
          frameUrl = undefined;
        }
      }

      setCameras((current) => {
        if (frameUrl && current[camera].frameUrl) {
          URL.revokeObjectURL(current[camera].frameUrl);
        }
        return {
          ...current,
          [camera]: {
            source: current[camera].source,
            running: status.running,
            fps: status.fps,
            plate: detected.latest?.plate_text || status.latest_plate || current[camera].plate,
            info: detected.latest?.info || status.latest_info || current[camera].info,
            frameUrl: frameUrl ?? current[camera].frameUrl,
            count: detected.count || status.detected || 0,
          },
        };
      });
    } catch {
      setCameras((current) => ({
        ...current,
        [camera]: { ...current[camera], running: false, fps: 0 },
      }));
    }
  }, []);

  useEffect(() => {
    const interval = window.setInterval(() => {
      void refreshCamera("cam1");
      void refreshCamera("cam2");
      void refreshCamera("cam3");
      void refreshCamera("cam4");
    }, 1200);
    return () => window.clearInterval(interval);
  }, [refreshCamera]);

  useEffect(() => {
    const timeout = window.setTimeout(() => void fetchData(), 0);
    const interval = window.setInterval(() => void fetchData(), 10000);
    return () => {
      window.clearTimeout(timeout);
      window.clearInterval(interval);
    };
  }, [fetchData]);

  useEffect(() => {
    return () => {
      Object.values(cameras).forEach((cam) => {
        if (cam.frameUrl) URL.revokeObjectURL(cam.frameUrl);
      });
    };
  }, [cameras]);

  const handleStart = async (camera: SmartCityCamera) => {
    try {
      await startSmartCityCamera(camera, cameras[camera].source);
      await refreshCamera(camera);
    } catch {
      console.error(`Failed to start camera ${camera}`);
    }
  };

  const handleStop = async (camera: SmartCityCamera) => {
    await stopSmartCityCamera(camera);
    setCameras((current) => ({
      ...current,
      [camera]: { ...current[camera], running: false, fps: 0, count: 0 },
    }));
  };

  // Count running cameras
  const runningCameras = Object.values(cameras).filter((c) => c.running).length;
  const totalDetections = Object.values(cameras).reduce((sum, c) => sum + c.count, 0);

  const flowByHour = useMemo(() => {
    const buckets = Array.from({ length: 24 }, (_, hour) => ({ hour, count: 0, violations: 0 }));
    vehicles.forEach((vehicle) => {
      buckets[getHour(vehicle.entry_time)].count += 1;
    });
    violations.forEach((violation) => {
      buckets[getHour(violation.timestamp)].violations += 1;
    });
    return buckets;
  }, [vehicles, violations]);

  const maxFlow = Math.max(1, ...flowByHour.map((item) => item.count));
  const peakHour = flowByHour.reduce((best, item) => item.count > best.count ? item : best, flowByHour[0]);

  const violationsByType = useMemo(() => {
    const map = new globalThis.Map<string, number>();
    violations.forEach((item) => map.set(item.violation_type, (map.get(item.violation_type) ?? 0) + 1));
    return Array.from(map.entries()).map(([type, count]) => ({ type, count })).sort((a, b) => b.count - a.count);
  }, [violations]);

  const cards = [
    { label: "Camera đang hoạt động", value: `${runningCameras}/4`, icon: Camera, color: "text-blue-400", bg: "bg-blue-500/10" },
    { label: "Phương tiện realtime", value: totalDetections, icon: Route, color: "text-cyan-400", bg: "bg-cyan-500/10" },
    { label: "Giờ cao điểm", value: `${peakHour.hour}:00`, icon: Clock, color: "text-emerald-400", bg: "bg-emerald-500/10" },
    { label: "Vi phạm ghi nhận", value: stats?.total_violations ?? violations.length, icon: AlertTriangle, color: "text-red-400", bg: "bg-red-500/10" },
  ];

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-500/10">
              <MapIcon className="h-5 w-5 text-blue-400" />
            </div>
            <div>
              <h2 className="font-bold text-white">Smart City - Quản lý đô thị & giao thông</h2>
              <p className="text-sm text-slate-400">4 luồng camera realtime, lưu lượng theo giờ, vi phạm và điểm cần can thiệp</p>
            </div>
          </div>
          <button onClick={() => void fetchData()} className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800">
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Làm mới dữ liệu thật
          </button>
        </div>
      </div>

      {/* 4 Camera Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <CameraMonitor
          camera="cam1"
          state={cameras.cam1}
          onStart={() => void handleStart("cam1")}
          onStop={() => void handleStop("cam1")}
          onRefresh={() => void refreshCamera("cam1")}
        />
        <CameraMonitor
          camera="cam2"
          state={cameras.cam2}
          onStart={() => void handleStart("cam2")}
          onStop={() => void handleStop("cam2")}
          onRefresh={() => void refreshCamera("cam2")}
        />
        <CameraMonitor
          camera="cam3"
          state={cameras.cam3}
          onStart={() => void handleStart("cam3")}
          onStop={() => void handleStop("cam3")}
          onRefresh={() => void refreshCamera("cam3")}
        />
        <CameraMonitor
          camera="cam4"
          state={cameras.cam4}
          onStart={() => void handleStart("cam4")}
          onStop={() => void handleStop("cam4")}
          onRefresh={() => void refreshCamera("cam4")}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {cards.map((card) => (
          <div key={card.label} className="card">
            <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${card.bg}`}>
              <card.icon className={`h-5 w-5 ${card.color}`} />
            </div>
            <p className="mt-4 text-3xl font-bold text-white">{card.value}</p>
            <p className="text-sm text-slate-400">{card.label}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1fr_420px]">
        <section className="card space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-white">Lưu lượng giao thông theo giờ</h3>
              <p className="text-sm text-slate-400">Tính từ thời gian vào của phương tiện trong cơ sở dữ liệu</p>
            </div>
            <BarChart3 className="h-5 w-5 text-cyan-400" />
          </div>
          <div className="space-y-2">
            {flowByHour.map((item) => (
              <div key={item.hour} className="grid grid-cols-[48px_1fr_80px] items-center gap-3 text-sm">
                <span className="text-slate-400">{item.hour.toString().padStart(2, "0")}:00</span>
                <div className="h-3 overflow-hidden rounded-full bg-slate-800">
                  <div className="h-full rounded-full bg-cyan-500" style={{ width: `${Math.max(2, (item.count / maxFlow) * 100)}%` }} />
                </div>
                <span className="text-right text-slate-300">{item.count} xe</span>
              </div>
            ))}
          </div>
        </section>

        <section className="card space-y-5">
          <div>
            <h3 className="font-semibold text-white">Cơ cấu vi phạm</h3>
            <p className="text-sm text-slate-400">Phân nhóm theo loại vi phạm đã ghi nhận</p>
          </div>
          {violationsByType.length === 0 ? (
            <div className="rounded-lg border border-slate-700 bg-slate-900 p-4 text-sm text-slate-400">Chưa có dữ liệu vi phạm</div>
          ) : violationsByType.map((item) => (
            <div key={item.type} className="rounded-lg border border-slate-700 bg-slate-900 p-3">
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-medium text-white">{violationLabel(item.type)}</span>
                <span className="badge bg-red-500/10 text-red-400">{item.count}</span>
              </div>
            </div>
          ))}
        </section>
      </div>

      <div className="card overflow-hidden p-0">
        <div className="flex items-center justify-between border-b border-slate-700/50 p-4">
          <div>
            <h3 className="font-semibold text-white">Sự kiện vi phạm mới nhất</h3>
            <p className="text-sm text-slate-400">Phát hiện đi sai quy định, vượt đèn đỏ, hành vi nguy hiểm</p>
          </div>
          <TrafficCone className="h-5 w-5 text-yellow-400" />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="p-3 text-left font-medium text-slate-400">Loại</th>
                <th className="p-3 text-left font-medium text-slate-400">Biển số</th>
                <th className="p-3 text-left font-medium text-slate-400">Phương tiện</th>
                <th className="p-3 text-left font-medium text-slate-400">Chi tiết</th>
                <th className="p-3 text-left font-medium text-slate-400">Thời gian</th>
              </tr>
            </thead>
            <tbody>
              {violations.length === 0 ? (
                <tr><td colSpan={5} className="p-8 text-center text-slate-500">Chưa có sự kiện vi phạm</td></tr>
              ) : violations.slice(0, 20).map((item) => (
                <tr key={item.id} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                  <td className="p-3"><span className="badge bg-red-500/10 text-red-400">{violationLabel(item.violation_type)}</span></td>
                  <td className="p-3 text-slate-300">{item.license_plate || "Chưa định danh"}</td>
                  <td className="p-3 text-slate-300">{item.vehicle_type || "-"}</td>
                  <td className="p-3 text-slate-300">{item.details}</td>
                  <td className="p-3 text-slate-300">{formatTime(item.timestamp)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}