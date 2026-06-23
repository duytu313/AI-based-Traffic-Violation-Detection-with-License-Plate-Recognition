"use client";

import type { Config } from "@/lib/types";

interface ConfigPanelProps {
  config: Config;
  onChange: (config: Config) => void;
}

export default function ConfigPanel({ config, onChange }: ConfigPanelProps) {
  const update = (partial: Partial<Config>) => {
    const next = { ...config, ...partial };
    // Ensure all required fields are present with defaults
    if (next.show_zones === undefined) next.show_zones = false;
    if (next.camera_direction === undefined) next.camera_direction = "down";
    onChange(next as Config);
  };

  // Use config directly with fallback values for rendering
  const c = {
    ...config,
    show_zones: config.show_zones ?? false,
    camera_direction: (config.camera_direction ?? "down") as "down" | "up",
  };
  return (
    <div className="card space-y-4">
      <h3 className="font-semibold text-white flex items-center gap-2">
        <span>⚙️ Cấu hình xử lý</span>
      </h3>

      {/* Violation Detection */}
      <div className="space-y-3">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={c.enable_violation_detection}
            onChange={(e) => update({ enable_violation_detection: e.target.checked })}
            className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500"
          />
          <span>Bật phát hiện vi phạm</span>
        </label>

        {c.enable_violation_detection && (
          <div className="space-y-2 pl-6">
            <div>
              <label className="text-xs text-slate-400">
                Ngưỡng vi phạm chung: {(c.violation_conf_limit * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={c.violation_conf_limit}
                onChange={(e) => update({ violation_conf_limit: parseFloat(e.target.value) })}
                className="w-full accent-blue-500"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400">
                Chở quá 2 người: {(c.conf_more_than_two * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={c.conf_more_than_two}
                onChange={(e) => update({ conf_more_than_two: parseFloat(e.target.value) })}
                className="w-full accent-blue-500"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400">
                Không mũ bảo hiểm: {(c.conf_no_helmet * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={c.conf_no_helmet}
                onChange={(e) => update({ conf_no_helmet: parseFloat(e.target.value) })}
                className="w-full accent-blue-500"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400">
                Dùng điện thoại: {(c.conf_using_mobile * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={c.conf_using_mobile}
                onChange={(e) => update({ conf_using_mobile: parseFloat(e.target.value) })}
                className="w-full accent-blue-500"
              />
            </div>
          </div>
        )}
      </div>

      {/* Red Light Detection */}
      <div className="space-y-3">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={c.enable_red_light_detection}
            onChange={(e) => update({ enable_red_light_detection: e.target.checked })}
            className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500"
          />
          <span>🚦 Bật phát hiện vượt đèn đỏ</span>
        </label>

        {c.enable_red_light_detection && (
          <div className="space-y-2 pl-6">
            <div>
              <label className="text-xs text-slate-400">
                Ngưỡng đèn: {(c.traffic_light_conf * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={c.traffic_light_conf}
                onChange={(e) => update({ traffic_light_conf: parseFloat(e.target.value) })}
                className="w-full accent-blue-500"
              />
            </div>
            <p className="text-xs text-slate-500">
              Vượt đèn đỏ được phát hiện khi xe nằm trong vùng ROI được vẽ thủ công.
            </p>
          </div>
        )}
      </div>

      {/* Zone Visualization */}
      <div className="space-y-3 pt-2 border-t border-slate-700">
        <div>
          <label className="text-xs text-slate-400">Hướng camera</label>
          <select
            value={c.camera_direction}
            onChange={(e) => update({ camera_direction: e.target.value as "down" | "up" })}
            className="w-full mt-1 px-3 py-1.5 text-sm bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="down">Camera trên cao (top-down)</option>
            <option value="up">Camera đằng sau (rear-view)</option>
          </select>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={c.show_zones === true}
            onChange={(e) => update({ show_zones: e.target.checked })}
            className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500"
          />
          <span>🗺️ Hiển thị vùng (Waiting/Stop/Intersection)</span>
        </label>
        {config.show_zones && (
          <div className="pl-6 space-y-1">
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-block w-3 h-3 bg-yellow-500/30 border border-yellow-500 rounded"></span>
              <span className="text-slate-400">Waiting Zone - Vùng chờ đèn</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-block w-3 h-3 bg-red-500/30 border border-red-500 rounded"></span>
              <span className="text-slate-400">Stop Zone - Vùng vạch dừng</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-block w-3 h-3 bg-green-500/30 border border-green-500 rounded"></span>
              <span className="text-slate-400">Intersection - Giao lộ</span>
            </div>
            <div className="flex items-center gap-2 text-xs mt-1">
              <span className="inline-block w-6 h-0.5 bg-red-500"></span>
              <span className="text-slate-400">Stop Line - Vạch dừng</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
