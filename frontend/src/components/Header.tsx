"use client";

import { Activity } from "lucide-react";
import Image from "next/image";

interface HeaderProps {
  isConnected: boolean;
}

export default function Header({ isConnected }: HeaderProps) {
  return (
    <header className="border-b border-slate-700/50 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl overflow-hidden bg-slate-800 flex items-center justify-center">
              <Image src="/image/logo.png" alt="Traffic AI" width={40} height={40} className="object-cover" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white leading-tight">
                Traffic AI
              </h1>
              <p className="text-xs text-slate-400">Smart Traffic Management System</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-sm">
              <Activity className="w-4 h-4" />
              <span className="text-slate-400">API:</span>
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  isConnected
                    ? "bg-green-500/10 text-green-400"
                    : "bg-red-500/10 text-red-400"
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    isConnected ? "bg-green-400" : "bg-red-400"
                  }`}
                />
                {isConnected ? "Connected" : "Disconnected"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}