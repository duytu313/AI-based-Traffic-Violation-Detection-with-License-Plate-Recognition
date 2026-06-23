"use client";

import {
  BarChart3,
  Camera,
  Car,
  Database,
  Factory,
  Image,
  Map,
  Settings2,
  Video,
} from "lucide-react";

export type TabId =
  | "image"
  | "video"
  | "webcam"
  | "dashboard"
  | "parking"
  | "logistics"
  | "smartcity"
  | "database"
  | "settings";

const tabs: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  { id: "parking", label: "Bãi đỗ xe", icon: Car },
  { id: "logistics", label: "Kho vận", icon: Factory },
  { id: "smartcity", label: "Smart City", icon: Map },
  { id: "image", label: "Xử lý ảnh", icon: Image },
  { id: "video", label: "Xử lý video", icon: Video },
  { id: "webcam", label: "Webcam / RTSP", icon: Camera },
  { id: "database", label: "Database", icon: Database },
  { id: "settings", label: "Cài đặt", icon: Settings2 },
];

interface TabsProps {
  active: TabId;
  onChange: (tab: TabId) => void;
}

export default function Tabs({ active, onChange }: TabsProps) {
  return (
    <nav className="flex w-52 shrink-0 flex-col gap-2">
      {tabs.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          className={`flex items-center gap-3 rounded-lg px-4 py-3 text-left font-medium transition-colors ${
            active === id
              ? "bg-blue-600 text-white"
              : "bg-slate-800 text-slate-300 hover:bg-slate-700"
          }`}
        >
          <Icon className="h-4 w-4 shrink-0" />
          <span className="text-sm">{label}</span>
        </button>
      ))}
    </nav>
  );
}