"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Camera,
  Car,
  CheckCircle2,
  Clock,
  Cpu,
  LogIn,
  LogOut,
  MapPin,
  RefreshCw,
  Search,
  ShieldCheck,
  Ticket,
  WalletCards,
} from "lucide-react";
import {
  getParkingGateDetected,
  getParkingGateFrame,
  getParkingGateStatus,
  startParkingGate,
  stopParkingGate,
  getParkingDbSlots,
  getParkingDbStats,
  getParkingDbEntries,
} from "@/lib/api";
import type { ParkingSlot, ParkingEntry, ParkingStats } from "@/lib/api";

type SlotStatus = "occupied" | "reserved" | "available" | "maintenance";
type VehicleType = "car" | "motorcycle" | "truck";
type GateType = "entry" | "exit";

type GateState = {
  source: string;
  plate: string;
  vehicleType: VehicleType;
  confidence: number;
  running: boolean;
  fps: number;
  status: "ready" | "processing" | "matched" | "warning";
  message: string;
  slot?: string;
  fee?: number;
  frameUrl?: string;
};

const statusMeta: Record<SlotStatus, { label: string; className: string; dot: string }> = {
  occupied: { label: "Đang đỗ", className: "border-blue-500/40 bg-blue-500/10 text-blue-300", dot: "bg-blue-400" },
  reserved: { label: "Đã đặt", className: "border-yellow-500/40 bg-yellow-500/10 text-yellow-300", dot: "bg-yellow-400" },
  available: { label: "Còn trống", className: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300", dot: "bg-emerald-400" },
  maintenance: { label: "Bảo trì", className: "border-slate-600 bg-slate-700/40 text-slate-400", dot: "bg-slate-500" },
};

const vehicleLabels: Record<string, string> = {
  car: "Ô tô",
  motorcycle: "Xe máy",
  truck: "Xe tải",
};

const initialGates: Record<GateType, GateState> = {
  entry: {
    source: "0",
    plate: "",
    vehicleType: "car",
    confidence: 0,
    running: false,
    fps: 0,
    status: "ready",
    message: "Sẵn sàng xử lý",
  },
  exit: {
    source: "1",
    plate: "",
    vehicleType: "car",
    confidence: 0,
    running: false,
    fps: 0,
    status: "ready",
    message: "Sẵn sàng xử lý",
  },
};

function formatCurrency(value: number) {
  return new Intl.NumberFormat("vi-VN", {
    style: "currency",
    currency: "VND",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatTime(value?: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
  });
}

function GateMonitor({
  type,
  state,
  onChange,
  onStart,
  onStop,
  onRefresh,
}: {
  type: GateType;
  state: GateState;
  onChange: (next: GateState) => void;
  onStart: () => void;
  onStop: () => void;
  onRefresh: () => void;
}) {
  const isEntry = type === "entry";
  const title = isEntry ? "Cổng vào" : "Cổng ra";
  const actionText = isEntry ? "Xác nhận cho vào" : "Xác nhận cho ra";
  const Icon = isEntry ? LogIn : LogOut;

  return (
    <section className="card space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${isEntry ? "bg-emerald-500/10" : "bg-blue-500/10"}`}>
            <Icon className={`h-5 w-5 ${isEntry ? "text-emerald-400" : "text-blue-400"}`} />
          </div>
          <div>
            <h3 className="font-semibold text-white">{title}</h3>
            <p className="text-sm text-slate-400">Xử lí biển số realtime</p>
          </div>
        </div>
        <span className={`${state.running ? "bg-emerald-500/10 text-emerald-400" : "bg-slate-700 text-slate-400"} badge`}>
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
        <div className="absolute inset-x-0 top-1/3 h-px bg-cyan-300/60 shadow-[0_0_18px_rgba(103,232,249,0.9)]" />
        <div className="absolute left-5 top-5 rounded border border-white/20 px-3 py-1 text-xs font-medium text-white/80">
          {title.toUpperCase()} CAM
        </div>
        <Camera className="absolute right-5 top-5 h-5 w-5 text-slate-300" />
        <div className="absolute bottom-5 left-5 right-5 rounded-lg border border-slate-600/80 bg-slate-950/80 p-3 backdrop-blur">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs text-slate-400">Biển số nhận diện</p>
              <p className="mt-1 text-2xl font-bold tracking-wide text-white">{state.plate || "Đang chờ..."}</p>
            </div>
            <div className="text-right">
              <p className="text-xs text-slate-400">Độ tin cậy</p>
              <p className="mt-1 text-lg font-semibold text-cyan-300">{state.confidence}%</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1.5">
          <span className="text-xs font-medium text-slate-400">Nguồn camera</span>
          <input
            value={state.source}
            onChange={(event) => onChange({ ...state, source: event.target.value })}
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>
        <label className="space-y-1.5">
          <span className="text-xs font-medium text-slate-400">Biển số</span>
          <input
            value={state.plate}
            onChange={(event) => onChange({ ...state, plate: event.target.value.toUpperCase() })}
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-slate-700 bg-slate-900 p-3">
          <p className="text-xs text-slate-500">Loại xe</p>
          <p className="mt-1 text-sm font-medium text-white">{vehicleLabels[state.vehicleType] || state.vehicleType}</p>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-900 p-3">
          <p className="text-xs text-slate-500">Ô liên quan</p>
          <p className="mt-1 text-sm font-medium text-white">{state.slot ?? "-"}</p>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-900 p-3">
          <p className="text-xs text-slate-500">Phí</p>
          <p className="mt-1 text-sm font-medium text-white">{formatCurrency(state.fee ?? 0)}</p>
        </div>
      </div>

      <div className={`rounded-lg border p-3 text-sm ${state.status === "warning" ? "border-red-500/30 bg-red-500/10 text-red-200" : "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"}`}>
        <div className="flex items-center gap-2 font-medium">
          {state.status === "warning" ? <AlertTriangle className="h-4 w-4" /> : <ShieldCheck className="h-4 w-4" />}
          {state.message}
        </div>
      </div>

      <div className="flex flex-col gap-2 sm:flex-row">
        <button
          onClick={onStart}
          className={`inline-flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-white transition-colors ${isEntry ? "bg-emerald-600 hover:bg-emerald-500" : "bg-blue-600 hover:bg-blue-500"}`}
        >
          <Icon className="h-4 w-4" />
          {state.running ? actionText : "Bắt đầu xử lí"}
        </button>
        <button onClick={onRefresh} className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800 transition-colors">
          <RefreshCw className="h-4 w-4" />
          Đọc lại
        </button>
        <button onClick={onStop} className="inline-flex items-center justify-center rounded-lg border border-slate-700 px-3 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800 transition-colors">
          Dừng
        </button>
      </div>
    </section>
  );
}

export default function ParkingManager() {
  const [query, setQuery] = useState("");
  const [selectedZone, setSelectedZone] = useState("all");
  const [selectedSlot, setSelectedSlot] = useState<ParkingSlot | null>(null);
  const [gates, setGates] = useState(initialGates);

  // Real data from parking database
  const [slots, setSlots] = useState<ParkingSlot[]>([]);
  const [entries, setEntries] = useState<ParkingEntry[]>([]);
  const [stats, setStats] = useState<ParkingStats | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchDbData = useCallback(async () => {
    try {
      const [slotRows, entryRows, statRows] = await Promise.all([
        getParkingDbSlots(),
        getParkingDbEntries(500, 0),
        getParkingDbStats(),
      ]);
      setSlots(slotRows);
      setEntries(entryRows);
      setStats(statRows);
    } catch (err) {
      console.error("Failed to fetch parking DB data:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshGate = useCallback(async (gate: GateType) => {
    try {
      const [status, detected] = await Promise.all([
        getParkingGateStatus(gate),
        getParkingGateDetected(gate),
      ]);
      let frameUrl: string | undefined;
      // Only try to get frame if camera is running
      if (status.running) {
        try {
          const blob = await getParkingGateFrame(gate);
          frameUrl = URL.createObjectURL(blob);
        } catch {
          frameUrl = undefined;
        }
      }

      setGates((current) => {
        if (frameUrl && current[gate].frameUrl) {
          URL.revokeObjectURL(current[gate].frameUrl);
        }
        const latestPlate = detected.latest?.plate_text || status.latest_plate || current[gate].plate;
        return {
          ...current,
          [gate]: {
            ...current[gate],
            plate: latestPlate,
            confidence: latestPlate ? current[gate].confidence || 95 : 0,
            running: status.running,
            fps: status.fps,
            frameUrl: frameUrl ?? current[gate].frameUrl,
            status: status.running ? "matched" : current[gate].status,
            message: status.running
              ? status.latest_info || current[gate].message
              : current[gate].message,
          },
        };
      });
    } catch {
      setGates((current) => ({
        ...current,
        [gate]: {
          ...current[gate],
          running: false,
          fps: 0,
        },
      }));
    }
  }, []);

  // Auto-refresh cameras
  useEffect(() => {
    const interval = window.setInterval(() => {
      void refreshGate("entry");
      void refreshGate("exit");
    }, 1200);
    return () => window.clearInterval(interval);
  }, [refreshGate]);

  // Auto-refresh DB data
  useEffect(() => {
    const timeout = window.setTimeout(() => void fetchDbData(), 0);
    const interval = window.setInterval(() => void fetchDbData(), 5000);
    return () => {
      window.clearTimeout(timeout);
      window.clearInterval(interval);
    };
  }, [fetchDbData]);

  useEffect(() => {
    return () => {
      Object.values(gates).forEach((gate) => {
        if (gate.frameUrl) URL.revokeObjectURL(gate.frameUrl);
      });
    };
  }, [gates]);

  const handleStart = async (gate: GateType) => {
    setGates((current) => ({
      ...current,
      [gate]: { ...current[gate], status: "processing", message: "Đang kết nối camera..." },
    }));
    try {
      await startParkingGate(gate, gates[gate].source);
      await refreshGate(gate);
    } catch {
      setGates((current) => ({
        ...current,
        [gate]: {
          ...current[gate],
          status: "warning",
          message: "Không mở được camera. Kiểm tra lại nguồn RTSP/webcam.",
        },
      }));
    }
  };

  const handleStop = async (gate: GateType) => {
    await stopParkingGate(gate);
    setGates((current) => ({
      ...current,
      [gate]: { ...current[gate], running: false, fps: 0, message: "Đã dừng luồng camera" },
    }));
  };

  const filteredSlots = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return slots.filter((slot) => {
      const matchesZone = selectedZone === "all" || slot.zone === selectedZone;
      const matchesQuery =
        !normalized ||
        slot.id.toLowerCase().includes(normalized) ||
        slot.current_plate?.toLowerCase().includes(normalized);
      return matchesZone && matchesQuery;
    });
  }, [slots, query, selectedZone]);

  const occupied = slots.filter((slot) => slot.status === "occupied").length;
  const available = slots.filter((slot) => slot.status === "available").length;
  const alerts = slots.filter((slot) => slot.status === "maintenance").length;
  const revenue = stats?.revenue || 0;
  const occupancyRate = slots.length > 0 ? Math.round((occupied / slots.length) * 100) : 0;
  const activeSessions = entries.filter((e) => e.status === "active");
  const zones = ["all", ...Array.from(new Set(slots.map((slot) => slot.zone)))];

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center">
              <Car className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h2 className="font-bold text-white">Quản lí bãi đỗ xe thông minh</h2>
              <p className="text-sm text-slate-400">Dữ liệu thật từ database parking.db - {slots.length} ô đỗ</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Cpu className="h-4 w-4 text-cyan-400" />
            ANPR realtime
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <GateMonitor
          type="entry"
          state={gates.entry}
          onChange={(next) => setGates((current) => ({ ...current, entry: next }))}
          onStart={() => void handleStart("entry")}
          onStop={() => void handleStop("entry")}
          onRefresh={() => void refreshGate("entry")}
        />
        <GateMonitor
          type="exit"
          state={gates.exit}
          onChange={(next) => setGates((current) => ({ ...current, exit: next }))}
          onStart={() => void handleStart("exit")}
          onStop={() => void handleStop("exit")}
          onRefresh={() => void refreshGate("exit")}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <div className="card">
          <div className="flex items-center justify-between">
            <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
              <Car className="w-5 h-5 text-blue-400" />
            </div>
            <span className="text-xs text-slate-500">{occupancyRate}% sử dụng</span>
          </div>
          <p className="mt-4 text-3xl font-bold text-white">{loading ? "..." : occupied}</p>
          <p className="text-sm text-slate-400">Xe đang gửi</p>
        </div>
        <div className="card">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          </div>
          <p className="mt-4 text-3xl font-bold text-white">{loading ? "..." : available}</p>
          <p className="text-sm text-slate-400">Ô còn trống</p>
        </div>
        <div className="card">
          <div className="w-10 h-10 rounded-lg bg-red-500/10 flex items-center justify-center">
            <AlertTriangle className="w-5 h-5 text-red-400" />
          </div>
          <p className="mt-4 text-3xl font-bold text-white">{loading ? "..." : alerts}</p>
          <p className="text-sm text-slate-400">Cảnh báo cần xử lí</p>
        </div>
        <div className="card">
          <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center">
            <WalletCards className="w-5 h-5 text-cyan-400" />
          </div>
          <p className="mt-4 text-3xl font-bold text-white">{loading ? "..." : formatCurrency(revenue)}</p>
          <p className="text-sm text-slate-400">Doanh thu tạm tính</p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1fr_360px]">
        <div className="card space-y-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h3 className="font-semibold text-white">Sơ đồ bãi đỗ</h3>
              <p className="text-sm text-slate-400">Dữ liệu thật từ database parking_slots</p>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Tìm ô hoặc biển số"
                  className="w-full rounded-lg border border-slate-700 bg-slate-900 py-2 pl-9 pr-3 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 sm:w-56"
                />
              </div>
              <select
                value={selectedZone}
                onChange={(event) => setSelectedZone(event.target.value)}
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {zones.map((zone) => (
                  <option key={zone} value={zone}>
                    {zone === "all" ? "Tất cả khu" : `Khu ${zone}`}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {filteredSlots.length === 0 ? (
              <div className="col-span-full py-8 text-center text-sm text-slate-500">
                {loading ? "Đang tải dữ liệu..." : "Không có ô đỗ nào"}
              </div>
            ) : filteredSlots.map((slot) => {
              const meta = statusMeta[slot.status as SlotStatus] || statusMeta.available;
              return (
                <button
                  key={slot.id}
                  onClick={() => setSelectedSlot(slot)}
                  className={`min-h-28 rounded-lg border p-3 text-left transition-colors ${meta.className} ${selectedSlot?.id === slot.id ? "ring-2 ring-blue-400" : "hover:border-white/30"}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-white">{slot.id}</span>
                    <span className={`h-2.5 w-2.5 rounded-full ${meta.dot}`} />
                  </div>
                  <p className="mt-3 text-xs font-medium">{meta.label}</p>
                  <p className="mt-1 min-h-5 truncate text-xs text-slate-300">{slot.current_plate ?? "-"}</p>
                </button>
              );
            })}
          </div>
        </div>

        <aside className="card space-y-5">
          {selectedSlot ? (
            <>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-white">Chi tiết ô {selectedSlot.id}</h3>
                  <p className="text-sm text-slate-400">Khu {selectedSlot.zone}</p>
                </div>
                <span className={`badge ${(statusMeta[selectedSlot.status as SlotStatus] || statusMeta.available).className}`}>
                  {(statusMeta[selectedSlot.status as SlotStatus] || statusMeta.available).label}
                </span>
              </div>
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
                  <span className="text-slate-400">Biển số</span>
                  <span className="font-medium text-white">{selectedSlot.current_plate ?? "Chưa có xe"}</span>
                </div>
                <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
                  <span className="text-slate-400">Loại xe</span>
                  <span className="text-slate-200">{selectedSlot.vehicle_type ? (vehicleLabels[selectedSlot.vehicle_type] || selectedSlot.vehicle_type) : "-"}</span>
                </div>
                <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
                  <span className="text-slate-400">Giờ vào</span>
                  <span className="text-slate-200">{formatTime(selectedSlot.entry_time)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Phí tạm tính</span>
                  <span className="font-semibold text-cyan-300">{formatCurrency(selectedSlot.fee || 0)}</span>
                </div>
              </div>
              <div className={`rounded-lg border p-3 text-sm ${selectedSlot.status === "maintenance" ? "border-red-500/30 bg-red-500/10 text-red-200" : "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"}`}>
                <div className="flex items-center gap-2 font-medium">
                  {selectedSlot.status === "maintenance" ? <AlertTriangle className="h-4 w-4" /> : <ShieldCheck className="h-4 w-4" />}
                  {selectedSlot.status === "maintenance" ? "Cần bảo trì" : "Trạng thái bình thường"}
                </div>
              </div>
            </>
          ) : (
            <div className="py-8 text-center text-sm text-slate-500">Chọn một ô để xem chi tiết</div>
          )}
        </aside>
      </div>

      <div className="card p-0 overflow-hidden">
        <div className="flex items-center justify-between border-b border-slate-700/50 p-4">
          <div>
            <h3 className="font-semibold text-white">Phiên gửi xe đang hoạt động</h3>
            <p className="text-sm text-slate-400">Dữ liệu thật từ database parking_entries</p>
          </div>
          <button onClick={() => void fetchDbData()} className="rounded-lg p-2 text-slate-400 hover:bg-slate-700 hover:text-white transition-colors">
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="p-3 text-left font-medium text-slate-400">Ô</th>
                <th className="p-3 text-left font-medium text-slate-400">Biển số</th>
                <th className="p-3 text-left font-medium text-slate-400">Loại xe</th>
                <th className="p-3 text-left font-medium text-slate-400">Giờ vào</th>
                <th className="p-3 text-left font-medium text-slate-400">Phí</th>
                <th className="p-3 text-left font-medium text-slate-400">Trạng thái</th>
              </tr>
            </thead>
            <tbody>
              {activeSessions.length === 0 ? (
                <tr><td colSpan={6} className="p-8 text-center text-slate-500">Chưa có xe đang gửi</td></tr>
              ) : activeSessions.map((entry) => (
                <tr key={entry.id} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                  <td className="p-3 text-slate-300"><span className="inline-flex items-center gap-1"><MapPin className="h-3 w-3 text-slate-500" />{entry.slot_id || "-"}</span></td>
                  <td className="p-3"><span className="badge bg-blue-500/10 text-blue-400"><Ticket className="h-3 w-3" />{entry.license_plate || "Không rõ"}</span></td>
                  <td className="p-3 text-slate-300">{entry.vehicle_type ? (vehicleLabels[entry.vehicle_type] || entry.vehicle_type) : "-"}</td>
                  <td className="p-3 text-slate-300"><span className="inline-flex items-center gap-1"><Clock className="h-3 w-3 text-slate-500" />{formatTime(entry.entry_time)}</span></td>
                  <td className="p-3 text-cyan-300">{formatCurrency(entry.fee || 0)}</td>
                  <td className="p-3">
                    {entry.fraud_alert ? (
                      <span className="badge bg-red-500/10 text-red-400">Gian lận</span>
                    ) : (
                      <span className="badge bg-emerald-500/10 text-emerald-400">Bình thường</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}