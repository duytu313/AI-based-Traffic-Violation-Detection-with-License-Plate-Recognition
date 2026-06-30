"use client";

import { useState, useEffect } from "react";
import {
  Database,
  Car,
  AlertTriangle,
  Search,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Clock,
} from "lucide-react";
import { getVehicles, getViolations, getFraudAlerts } from "@/lib/api";
import type { DBVehicle, DBViolation } from "@/lib/types";

type ViewType = "vehicles" | "violations" | "fraud";

export default function DatabaseView() {
  const [view, setView] = useState<ViewType>("vehicles");
  const [vehicles, setVehicles] = useState<DBVehicle[]>([]);
  const [violations, setViolations] = useState<DBViolation[]>([]);
  const [fraudAlerts, setFraudAlerts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchPlate, setSearchPlate] = useState("");
  const [page, setPage] = useState(0);
  const [totalCount, setTotalCount] = useState(0);
  const limit = 20;

  useEffect(() => {
    fetchData();
  }, [view, page, searchPlate]);

  const fetchData = async () => {
    setLoading(true);
    try {
      if (view === "vehicles") {
        const data = await getVehicles({
          limit,
          offset: page * limit,
          license_plate: searchPlate || undefined,
        });
        setVehicles(data);
        setTotalCount(data.length);
      } else if (view === "violations") {
        const data = await getViolations({
          limit,
          offset: page * limit,
        });
        setViolations(data);
      } else if (view === "fraud") {
        const data = await getFraudAlerts(limit);
        setFraudAlerts(data);
      }
    } catch {
      // silent
    } finally {
      setLoading(false);
    }
  };

  const getViolationBadge = (type: string) => {
    const colors: Record<string, string> = {
      WITHOUT_HELMET: "bg-orange-500/10 text-orange-400",
      MORE_THAN_TWO_PERSONS: "bg-red-500/10 text-red-400",
      USING_MOBILE: "bg-purple-500/10 text-purple-400",
      RED_LIGHT_VIOLATION: "bg-red-500/10 text-red-400",
    };
    return colors[type] || "bg-slate-500/10 text-slate-400";
  };

  const formatTime = (t: string) => {
    try {
      return new Date(t).toLocaleString("vi-VN");
    } catch {
      return t;
    }
  };

  const tabs: { id: ViewType; label: string; icon: React.ReactNode }[] = [
    { id: "vehicles", label: "Vehicles", icon: <Car className="w-4 h-4" /> },
    { id: "violations", label: "Violations", icon: <AlertTriangle className="w-4 h-4" /> },
    { id: "fraud", label: "Fraud", icon: <AlertTriangle className="w-4 h-4" /> },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
              <Database className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <h2 className="font-bold text-white">Database</h2>
              <p className="text-sm text-slate-400">Vehicle and violation history</p>
            </div>
          </div>
          <button
            onClick={fetchData}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <RefreshCw className={`w-4 h-4 text-slate-400 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {/* Sub Tabs */}
      <div className="flex gap-2">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => {
              setView(t.id);
              setPage(0);
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              view === t.id
                ? "bg-blue-500/10 text-blue-400 border border-blue-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
            }`}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {/* Search */}
      {view === "vehicles" && (
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search license plate..."
            value={searchPlate}
            onChange={(e) => {
              setSearchPlate(e.target.value);
              setPage(0);
            }}
            className="w-full pl-10 pr-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      )}

      {/* Table */}
      <div className="card p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/50">
                {view === "vehicles" && (
                  <>
                    <th className="text-left p-3 text-slate-400 font-medium">ID</th>
                    <th className="text-left p-3 text-slate-400 font-medium">License Plate</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Type</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Color</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Entry</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Exit</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Fraud</th>
                  </>
                )}
                {view === "violations" && (
                  <>
                    <th className="text-left p-3 text-slate-400 font-medium">ID</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Type</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Details</th>
                    <th className="text-left p-3 text-slate-400 font-medium">License Plate</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Time</th>
                  </>
                )}
                {view === "fraud" && (
                  <>
                    <th className="text-left p-3 text-slate-400 font-medium">ID</th>
                    <th className="text-left p-3 text-slate-400 font-medium">License Plate</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Vehicle Type</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Reason</th>
                    <th className="text-left p-3 text-slate-400 font-medium">Time</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} className="p-8 text-center text-slate-500">
                    <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2" />
                    Loading...
                  </td>
                </tr>
              ) : view === "vehicles" && vehicles.length === 0 ? (
                <tr>
                  <td colSpan={7} className="p-8 text-center text-slate-500">
                    No data available
                  </td>
                </tr>
              ) : view === "violations" && violations.length === 0 ? (
                <tr>
                  <td colSpan={5} className="p-8 text-center text-slate-500">
                    No data available
                  </td>
                </tr>
              ) : view === "fraud" && fraudAlerts.length === 0 ? (
                <tr>
                  <td colSpan={5} className="p-8 text-center text-slate-500">
                    No data available
                  </td>
                </tr>
              ) : (
                <>
                  {view === "vehicles" &&
                    vehicles.map((v) => (
                      <tr
                        key={v.id}
                        className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors"
                      >
                        <td className="p-3 text-slate-400">{v.id}</td>
                        <td className="p-3">
                          {v.license_plate ? (
                            <span className="badge bg-blue-500/10 text-blue-400">
                              {v.license_plate}
                            </span>
                          ) : (
                            <span className="text-slate-500">N/A</span>
                          )}
                        </td>
                        <td className="p-3 text-slate-300">{v.vehicle_type}</td>
                        <td className="p-3 text-slate-300">{v.color}</td>
                        <td className="p-3 text-slate-300 text-xs">
                          <div className="flex items-center gap-1">
                            <Clock className="w-3 h-3 text-slate-500" />
                            {formatTime(v.entry_time)}
                          </div>
                        </td>
                        <td className="p-3 text-slate-300 text-xs">
                          {v.exit_time ? formatTime(v.exit_time) : "-"}
                        </td>
                        <td className="p-3">
                          {v.fraud_alert ? (
                            <span className="badge bg-red-500/10 text-red-400">Alert</span>
                          ) : (
                            <span className="text-slate-500">-</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  {view === "violations" &&
                    violations.map((v) => (
                      <tr
                        key={v.id}
                        className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors"
                      >
                        <td className="p-3 text-slate-400">{v.id}</td>
                        <td className="p-3">
                          <span
                            className={`badge ${getViolationBadge(v.violation_type)}`}
                          >
                            {v.violation_type}
                          </span>
                        </td>
                        <td className="p-3 text-slate-300 text-sm">{v.details}</td>
                        <td className="p-3">
                          {v.license_plate ? (
                            <span className="badge bg-blue-500/10 text-blue-400">
                              {v.license_plate}
                            </span>
                          ) : (
                            <span className="text-slate-500">N/A</span>
                          )}
                        </td>
                        <td className="p-3 text-slate-300 text-xs">
                          {formatTime(v.timestamp)}
                        </td>
                      </tr>
                    ))}
                  {view === "fraud" &&
                    fraudAlerts.map((f: any) => (
                      <tr
                        key={f.id}
                        className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors"
                      >
                        <td className="p-3 text-slate-400">{f.id}</td>
                        <td className="p-3">
                          <span className="badge bg-blue-500/10 text-blue-400">
                            {f.license_plate || "N/A"}
                          </span>
                        </td>
                        <td className="p-3 text-slate-300">{f.vehicle_type}</td>
                        <td className="p-3 text-red-300 text-sm">{f.fraud_reason}</td>
                        <td className="p-3 text-slate-300 text-xs">
                          {formatTime(f.entry_time)}
                        </td>
                      </tr>
                    ))}
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {view === "vehicles" && (
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={() => setPage(Math.max(0, page - 1))}
            disabled={page === 0}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft className="w-4 h-4 text-slate-300" />
          </button>
          <span className="text-sm text-slate-400">
            Page {page + 1}
          </span>
          <button
            onClick={() => setPage(page + 1)}
            disabled={vehicles.length < limit}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronRight className="w-4 h-4 text-slate-300" />
          </button>
        </div>
      )}
    </div>
  );
}