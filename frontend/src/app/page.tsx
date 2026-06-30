"use client";

import { useEffect, useState } from "react";
import Header from "@/components/Header";
import Tabs, { type TabId } from "@/components/Tabs";
import ImageProcessor from "@/components/ImageProcessor";
import VideoProcessor from "@/components/VideoProcessor";
import WebcamProcessor from "@/components/WebcamProcessor";
import Dashboard from "@/components/Dashboard";
import ParkingManager from "@/components/ParkingManager";
import LogisticsOperations from "@/components/LogisticsOperations";
import SmartCityDashboard from "@/components/SmartCityDashboard";
import DatabaseView from "@/components/DatabaseView";
import SettingsComponent from "@/components/Settings";
import { healthCheck } from "@/lib/api";

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("dashboard");
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const check = async () => {
      const ok = await healthCheck();
      setIsConnected(ok);
    };
    void check();
    const interval = setInterval(check, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-900">
      <Header isConnected={isConnected} />
      <div className="flex">
        <Tabs active={activeTab} onChange={setActiveTab} />
        <main className="mx-auto flex-1 px-4 py-8 sm:px-6 lg:px-8">
          {!isConnected && (
            <div className="mb-6 flex items-start gap-3 rounded-lg border border-yellow-500/20 bg-yellow-500/10 p-4">
              <div className="mt-2 h-2 w-2 shrink-0 animate-pulse rounded-full bg-yellow-400" />
              <div>
                <p className="text-sm font-medium text-yellow-300">Waiting for API server connection...</p>
                <p className="mt-1 text-xs text-yellow-400/60">
                  Please ensure the backend server is running at {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
                </p>
              </div>
            </div>
          )}

          {activeTab === "dashboard" && <Dashboard />}
          {activeTab === "parking" && <ParkingManager />}
          {activeTab === "logistics" && <LogisticsOperations />}
          {activeTab === "smartcity" && <SmartCityDashboard />}
          {activeTab === "image" && <ImageProcessor />}
          {activeTab === "video" && <VideoProcessor />}
          {activeTab === "webcam" && <WebcamProcessor />}
          {activeTab === "database" && <DatabaseView />}
          {activeTab === "settings" && <SettingsComponent />}
        </main>
      </div>
    </div>
  );
}