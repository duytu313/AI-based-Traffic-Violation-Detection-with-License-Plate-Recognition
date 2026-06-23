"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Camera,
  Construction,
  Factory,
  LogIn,
  RefreshCw,
  Search,
  ShieldCheck,
  Truck,
  UserCheck,
  XCircle,
} from "lucide-react";
import {
  getFraudAlerts,
  getVehicles,
  getLogisticsCameraDetected,
  getLogisticsCameraFrame,
  getLogisticsCameraStatus,
  startLogisticsCamera,
  stopLogisticsCamera,
} from "@/lib/api";
import type { DBVehicle } from "@/lib/types";
import type {
  LogisticsCamera,
} from "@/lib/api";

type FraudAlert = {
  id: number;
  license_plate: string | null;
  vehicle_type: string | null;
  color: string | null;
  entry_time: string;
  fraud_reason: string | null;
};

type UnknownVehicleAlert = {
  time: number;
  track_id: number;
  vehicle_type: string;
  color: string;
  message: string;
};

type CameraState = {
  source: string;
  running: boolean;
  fps: number;
  plate: string;
  info: string;
  frameUrl?: string;
  unknownAlerts: UnknownVehicleAlert[];
};

function formatTime(value?: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleString("vi-VN");
}

function diffHours(from?: string | null, to?: string | null) {
  if (!from) return 0;
  const end = to ? new Date(to).getTime() : Date.now();
  return Math.max(0, Math.round((end - new Date(from).getTime()) / 36_000) / 10);
}

function isUnknownPlate(plate?: string | null) {
  const normalized = (plate ?? "").trim().toUpperCase();
  return !normalized || normalized === "N/A" || normalized === "UNKNOWN";
}

const cameraLabels: Record<LogisticsCamera, { label: string; icon: React.ElementType; color: string }> = {
  gate: { label: "Cổng ra vào", icon: LogIn, color: "text-emerald-400" },
  construction_site: { label: "Công trường", icon: Construction, color: "text-yellow-400" },
};

function CameraMonitor({
  camera,
  state,
  onStart,
  onStop,
  onRefresh,
}: {
  camera: LogisticsCamera;
  state: CameraState;
  onStart: () => void;
  onStop: () => void;
  onRefresh: () => void;
}) {
  const meta = cameraLabels[camera];
  const Icon = meta.icon;

  return (
    <section className="card space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-700">
            <Icon className={`h-5 w-5 ${meta.color}`} />
          </div>
          <div>
            <h3 className="font-semibold text-white">{meta.label}</h3>
            <p className="text-sm text-slate-400">Phát hiện xe lạ realtime</p>
          </div>
        </div>
        <span className={`badge ${state.running ? "bg-emerald-500/10 text-emerald-400" : "bg-slate-700 text-slate-400"}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${state.running ? "bg-emerald-400" : "bg-slate-500"}`} />
          {state.running ? `Live ${state.fps.toFixed(1)} FPS` : "Tạm dừng"}
        </span>
      </div>

      <div className="relative aspect-video overflow-hidden rounded-lg border border-slate-700 bg-slate-950">
        {state.frameUrl ? (
          <div className="absolute inset-0 bg-cover bg-center" style={{ backgroundImage: `url(${state.frameUrl})` }} />
        ) : (
          <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(30,41,59,0.9),rgba(2,6,23,0.95))]" />
        )}
        <div className="absolute inset-0 bg-slate-950/25" />
        <div className="absolute left-5 top-5 rounded border border-white/20 px-3 py-1 text-xs font-medium text-white/80">
          {camera === "gate" ? "GATE CAM" : "SITE CAM"}
        </div>
        <Camera className="absolute right-5 top-5 h-5 w-5 text-slate-300" />
        <div className="absolute bottom-5 left-5 right-5 rounded-lg border border-slate-600/80 bg-slate-950/80 p-3 backdrop-blur">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs text-slate-400">Biển số nhận diện</p>
              <p className="mt-1 text-2xl font-bold tracking-wide text-white">{state.plate || "Đang chờ..."}</p>
            </div>
            <div className="text-right">
              <p className="text-xs text-slate-400">Xe lạ</p>
              <p className="mt-1 text-lg font-semibold text-yellow-300">{state.unknownAlerts.length}</p>
            </div>
          </div>
        </div>
      </div>

      <label className="space-y-1.5">
        <span className="text-xs font-medium text-slate-400">Nguồn camera (RTSP/webcam)</span>
        <input
          value={state.source}
          onChange={() => {}}
          placeholder="0 (webcam) hoặc rtsp://..."
          className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </label>

      {/* Unknown vehicle alerts */}
      {state.unknownAlerts.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-yellow-400">Cảnh báo xe lạ gần đây:</p>
          <div className="max-h-32 space-y-1.5 overflow-y-auto">
            {state.unknownAlerts.slice(-5).reverse().map((alert, idx) => (
              <div key={idx} className="flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/5 p-2">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-400" />
                <div className="text-xs">
                  <p className="text-red-200">{alert.message}</p>
                  <p className="text-slate-500">ID:{alert.track_id} - {new Date(alert.time * 1000).toLocaleTimeString("vi-VN")}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-col gap-2 sm:flex-row">
        <button
          onClick={onStart}
          className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-500"
        >
          <Camera className="h-4 w-4" />
          {state.running ? "Đang chạy" : "Bắt đầu"}
        </button>
        <button
          onClick={onRefresh}
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm font-medium text-slate-300 transition-colors hover:bg-slate-800"
        >
          <RefreshCw className="h-4 w-4" />
          Làm mới
        </button>
        <button
          onClick={onStop}
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-red-700 px-3 py-2 text-sm font-medium text-red-300 transition-colors hover:bg-red-800"
        >
          <XCircle className="h-4 w-4" />
          Dừng
        </button>
      </div>
    </section>
  );
}

export default function LogisticsOperations() {
  const [vehicles, setVehicles] = useState<DBVehicle[]>([]);
  const [fraudAlerts, setFraudAlerts] = useState<FraudAlert[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);

  // Camera states
  const [cameras, setCameras] = useState<Record<LogisticsCamera, CameraState>>({
    gate: { source: "0", running: false, fps: 0, plate: "", info: "", unknownAlerts: [] },
    construction_site: { source: "1", running: false, fps: 0, plate: "", info: "", unknownAlerts: [] },
  });

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [vehicleRows, fraudRows] = await Promise.all([
        getVehicles({ limit: 500, offset: 0 }),
        getFraudAlerts(200) as Promise<FraudAlert[]>,
      ]);
      setVehicles(vehicleRows);
      setFraudAlerts(fraudRows);
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-refresh cameras
  const refreshCamera = useCallback(async (camera: LogisticsCamera) => {
    try {
      const [status, detected] = await Promise.all([
        getLogisticsCameraStatus(camera),
        getLogisticsCameraDetected(camera),
      ]);
      let frameUrl: string | undefined;
      // Only try to get frame if camera is running
      if (status.running) {
        try {
          const blob = await getLogisticsCameraFrame(camera);
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
            unknownAlerts: detected.unknown_vehicle_alerts || [],
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
      void refreshCamera("gate");
      void refreshCamera("construction_site");
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

  const handleStart = async (camera: LogisticsCamera) => {
    try {
      await startLogisticsCamera(camera, cameras[camera].source);
      await refreshCamera(camera);
    } catch {
      console.error(`Failed to start camera ${camera}`);
    }
  };

  const handleStop = async (camera: LogisticsCamera) => {
    await stopLogisticsCamera(camera);
    setCameras((current) => ({
      ...current,
      [camera]: { ...current[camera], running: false, fps: 0, unknownAlerts: [] },
    }));
  };

  const activeInside = useMemo(() => vehicles.filter((v) => !v.exit_time), [vehicles]);
  const truckVisits = useMemo(() => vehicles.filter((v) => (v.vehicle_type ?? "").toLowerCase().includes("truck")), [vehicles]);
  const unknownPlateRows = useMemo(() => vehicles.filter((v) => isUnknownPlate(v.license_plate)), [vehicles]);
  const longStayRows = useMemo(() => activeInside.filter((v) => diffHours(v.entry_time) >= 4), [activeInside]);

  const filteredActive = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return activeInside.filter((v) => {
      if (!normalized) return true;
      return [v.license_plate, v.vehicle_type, v.color, String(v.track_id ?? "")]
        .join(" ")
        .toLowerCase()
        .includes(normalized);
    });
  }, [activeInside, query]);

  // Count total unknown alerts from both cameras
  const totalUnknownAlerts = cameras.gate.unknownAlerts.length + cameras.construction_site.unknownAlerts.length;

  const riskRows = useMemo(() => {
    const rows = [...fraudAlerts.map((item) => ({
      id: `fraud-${item.id}`,
      plate: item.license_plate || "Không rõ",
      type: item.vehicle_type || "-",
      reason: item.fraud_reason || "Nghi vấn gian lận dữ liệu xe ra vào",
      time: item.entry_time,
      level: "Cao",
    }))];

    unknownPlateRows.slice(0, 8).forEach((item) => rows.push({
      id: `unknown-${item.id}`,
      plate: item.license_plate || "Không đọc được",
      type: item.vehicle_type || "-",
      reason: "OCR không đọc được biển số, cần bảo vệ xác minh trước khi cho xe qua cổng",
      time: item.entry_time,
      level: "Trung bình",
    }));

    longStayRows.slice(0, 8).forEach((item) => rows.push({
      id: `long-${item.id}`,
      plate: item.license_plate || "Không rõ",
      type: item.vehicle_type || "-",
      reason: `Xe còn trong khu ${diffHours(item.entry_time)} giờ, cần đối chiếu lệnh xuất/nhập hàng`,
      time: item.entry_time,
      level: "Theo dõi",
    }));

    return rows;
  }, [fraudAlerts, longStayRows, unknownPlateRows]);

  const cards = [
    { label: "Xe đang trong khu", value: activeInside.length, icon: Truck, color: "text-blue-400", bg: "bg-blue-500/10" },
    { label: "Lượt xe tải ghi nhận", value: truckVisits.length, icon: Factory, color: "text-cyan-400", bg: "bg-cyan-500/10" },
    { label: "Biển số cần xác minh", value: unknownPlateRows.length, icon: UserCheck, color: "text-yellow-400", bg: "bg-yellow-500/10" },
    { label: "Cảnh báo xe lạ (realtime)", value: totalUnknownAlerts, icon: AlertTriangle, color: "text-red-400", bg: "bg-red-500/10" },
  ];

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500/10">
              <Factory className="h-5 w-5 text-cyan-400" />
            </div>
            <div>
              <h2 className="font-bold text-white">Khu công nghiệp & Kho vận</h2>
              <p className="text-sm text-slate-400">2 luồng camera realtime: cổng ra vào + công trường - phát hiện xe lạ, kiểm soát xe tải</p>
            </div>
          </div>
          <button onClick={() => void fetchData()} className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800">
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Làm mới dữ liệu thật
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <CameraMonitor
          camera="gate"
          state={cameras.gate}
          onStart={() => void handleStart("gate")}
          onStop={() => void handleStop("gate")}
          onRefresh={() => void refreshCamera("gate")}
        />
        <CameraMonitor
          camera="construction_site"
          state={cameras.construction_site}
          onStart={() => void handleStart("construction_site")}
          onStop={() => void handleStop("construction_site")}
          onRefresh={() => void refreshCamera("construction_site")}
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
        <section className="card overflow-hidden p-0">
          <div className="flex flex-col gap-3 border-b border-slate-700/50 p-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h3 className="font-semibold text-white">Xe đang ở trong khu</h3>
              <p className="text-sm text-slate-400">Dữ liệu lấy từ bảng vehicles: xe chưa có thời gian ra</p>
            </div>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Tìm biển số, loại xe..."
                className="w-full rounded-lg border border-slate-700 bg-slate-900 py-2 pl-9 pr-3 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 lg:w-64"
              />
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700/50">
                  <th className="p-3 text-left font-medium text-slate-400">Biển số</th>
                  <th className="p-3 text-left font-medium text-slate-400">Loại xe</th>
                  <th className="p-3 text-left font-medium text-slate-400">Màu</th>
                  <th className="p-3 text-left font-medium text-slate-400">Giờ vào</th>
                  <th className="p-3 text-left font-medium text-slate-400">Thời gian lưu</th>
                  <th className="p-3 text-left font-medium text-slate-400">Trạng thái</th>
                </tr>
              </thead>
              <tbody>
                {filteredActive.length === 0 ? (
                  <tr><td colSpan={6} className="p-8 text-center text-slate-500">Chưa có xe đang trong khu</td></tr>
                ) : filteredActive.map((v) => (
                  <tr key={v.id} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                    <td className="p-3"><span className="badge bg-blue-500/10 text-blue-400">{v.license_plate || "Không đọc được"}</span></td>
                    <td className="p-3 text-slate-300">{v.vehicle_type || "-"}</td>
                    <td className="p-3 text-slate-300">{v.color || "-"}</td>
                    <td className="p-3 text-slate-300">{formatTime(v.entry_time)}</td>
                    <td className="p-3 text-slate-300">{diffHours(v.entry_time)} giờ</td>
                    <td className="p-3">
                      {isUnknownPlate(v.license_plate) ? <span className="badge bg-yellow-500/10 text-yellow-400">Cần xác minh</span> : <span className="badge bg-emerald-500/10 text-emerald-400">Hợp lệ</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-white">Hàng đợi rủi ro</h3>
              <p className="text-sm text-slate-400">Gian lận, xe lạ, lưu khu bất thường</p>
            </div>
            <ShieldCheck className="h-5 w-5 text-cyan-400" />
          </div>
          <div className="space-y-3">
            {riskRows.length === 0 ? (
              <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-200">Không có cảnh báo rủi ro từ dữ liệu hiện tại</div>
            ) : riskRows.slice(0, 10).map((item) => (
              <div key={item.id} className="rounded-lg border border-slate-700 bg-slate-900 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-medium text-white">{item.plate}</p>
                    <p className="mt-1 text-xs text-slate-400">{item.type} • {formatTime(item.time)}</p>
                  </div>
                  <span className={`badge ${item.level === "Cao" ? "bg-red-500/10 text-red-400" : "bg-yellow-500/10 text-yellow-400"}`}>{item.level}</span>
                </div>
                <p className="mt-2 text-sm text-slate-300">{item.reason}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}