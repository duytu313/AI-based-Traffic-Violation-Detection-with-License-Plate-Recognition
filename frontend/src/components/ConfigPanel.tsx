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
    if (next.show_bev === undefined) next.show_bev = true;
    if (next.enable_bev_detection === undefined) next.enable_bev_detection = true;
    if (next.camera_direction === undefined) next.camera_direction = "down";
    onChange(next as Config);
  };

  // Use config directly with fallback values for rendering
  const c = {
    ...config,
    show_zones: config.show_zones ?? false,
    show_bev: config.show_bev ?? true,
    enable_bev_detection: config.enable_bev_detection ?? true,
    camera_direction: (config.camera_direction ?? "down") as "down" | "up",
  };
  return (
    <div className="card space-y-4">
      <h3 className="font-semibold text-white flex items-center gap-2">
        <span>⚙️ Processing Configuration</span>
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
          <span>Enable violation detection</span>
        </label>

        {c.enable_violation_detection && (
          <div className="space-y-2 pl-6">
            <div>
              <label className="text-xs text-slate-400">
                General violation threshold: {(c.violation_conf_limit * 100).toFixed(0)}%
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
                Carrying more than 2 people: {(c.conf_more_than_two * 100).toFixed(0)}%
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
                No helmet: {(c.conf_no_helmet * 100).toFixed(0)}%
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
                Using phone: {(c.conf_using_mobile * 100).toFixed(0)}%
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
          <span>🚦 Enable red light violation detection</span>
        </label>

        {c.enable_red_light_detection && (
          <div className="space-y-2 pl-6">
            <div>
              <label className="text-xs text-slate-400">
                Light threshold: {(c.traffic_light_conf * 100).toFixed(0)}%
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
              Red light violations are detected when vehicles are inside the manually drawn ROI zone.
            </p>
          </div>
        )}
      </div>

      {/* Zone Visualization */}
      <div className="space-y-3 pt-2 border-t border-slate-700">
        <div>
          <label className="text-xs text-slate-400">Camera direction</label>
          <select
            value={c.camera_direction}
            onChange={(e) => update({ camera_direction: e.target.value as "down" | "up" })}
            className="w-full mt-1 px-3 py-1.5 text-sm bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="down">Overhead camera (top-down)</option>
            <option value="up">Rear camera (rear-view)</option>
          </select>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={c.show_zones === true}
            onChange={(e) => update({ show_zones: e.target.checked })}
            className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500"
          />
          <span>🗺️ Show zones (Waiting/Stop/Intersection)</span>
        </label>
        {config.show_zones && (
          <div className="pl-6 space-y-1">
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-block w-3 h-3 bg-yellow-500/30 border border-yellow-500 rounded"></span>
               <span className="text-slate-400">Waiting Zone - Red light waiting area</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-block w-3 h-3 bg-red-500/30 border border-red-500 rounded"></span>
               <span className="text-slate-400">Stop Zone - Stop line area</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-block w-3 h-3 bg-green-500/30 border border-green-500 rounded"></span>
               <span className="text-slate-400">Intersection - Intersection area</span>
            </div>
            <div className="flex items-center gap-2 text-xs mt-1">
              <span className="inline-block w-6 h-0.5 bg-red-500"></span>
               <span className="text-slate-400">Stop Line - Stop line</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
