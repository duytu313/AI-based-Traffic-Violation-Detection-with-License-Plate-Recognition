"use client";

import { useState, useEffect } from "react";
import { BarChart3, Car, AlertTriangle, Shield, TrendingUp } from "lucide-react";
import { getStats } from "@/lib/api";
import type { Stats } from "@/lib/types";

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const data = await getStats();
        setStats(data);
      } catch {
        // silent
      } finally {
        setLoading(false);
      }
    };
    fetch();
    const interval = setInterval(fetch, 10000);
    return () => clearInterval(interval);
  }, []);

  const cards = [
    {
      label: "Total Vehicles",
      value: stats?.total_vehicles ?? 0,
      icon: Car,
      color: "text-blue-400",
      bg: "bg-blue-500/10",
    },
    {
      label: "Fraud Alerts",
      value: stats?.fraud_alerts ?? 0,
      icon: Shield,
      color: "text-yellow-400",
      bg: "bg-yellow-500/10",
    },
    {
      label: "Violations",
      value: stats?.total_violations ?? 0,
      icon: AlertTriangle,
      color: "text-red-400",
      bg: "bg-red-500/10",
    },
  ];

  return (
    <div className="space-y-8">
      {/* Welcome */}
      <div className="card">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
            <BarChart3 className="w-7 h-7 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">
              Dashboard
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Overview of the smart traffic management system
            </p>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-8 w-20 bg-slate-700 rounded mb-3" />
              <div className="h-4 w-32 bg-slate-700 rounded" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {cards.map((card) => (
            <div key={card.label} className="card">
              <div className="flex items-center justify-between mb-3">
                <div className={`w-10 h-10 rounded-lg ${card.bg} flex items-center justify-center`}>
                  <card.icon className={`w-5 h-5 ${card.color}`} />
                </div>
                <TrendingUp className="w-4 h-4 text-slate-500" />
              </div>
              <p className="text-3xl font-bold text-white">{card.value}</p>
              <p className="text-sm text-slate-400 mt-1">{card.label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="font-semibold text-white mb-3">🚀 User Guide</h3>
          <ol className="space-y-2 text-sm text-slate-300">
            <li>1. Select <strong>Image Processing</strong> tab to upload and analyze images</li>
            <li>2. Select <strong>Video Processing</strong> tab to analyze videos</li>
            <li>3. Select <strong>Database</strong> tab to view history</li>
            <li>4. Adjust detection thresholds in the <strong>Configuration</strong> tab</li>
          </ol>
        </div>
        <div className="card">
          <h3 className="font-semibold text-white mb-3">⚡ Features</h3>
          <ul className="space-y-2 text-sm text-slate-300">
            <li>✅ Vehicle detection (cars, motorcycles, buses, trucks)</li>
            <li>✅ Automatic license plate recognition (ANPR)</li>
            <li>✅ Violation detection: no helmet, carrying more than 2 people, using phone</li>
            <li>✅ Red light violation detection</li>
            <li>✅ Telegram notifications</li>
            <li>✅ Database storage</li>
          </ul>
        </div>
      </div>
    </div>
  );
}