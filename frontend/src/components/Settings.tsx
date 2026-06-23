"use client";

import { useState } from "react";
import { Settings2, Bell, Send, CheckCircle2, AlertTriangle } from "lucide-react";

export default function Settings() {
  const [botToken, setBotToken] = useState("");
  const [chatId, setChatId] = useState("");
  const [testResult, setTestResult] = useState<{ success: boolean; msg: string } | null>(null);

  const handleTestTelegram = async () => {
    setTestResult(null);
    try {
      const url = `https://api.telegram.org/bot${botToken}/sendMessage`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chat_id: chatId,
          text: "✅ Kết nối thành công! Bot sẽ gửi ảnh khi phát hiện xe mới.",
        }),
      });
      const data = await res.json();
      if (data.ok) {
        setTestResult({ success: true, msg: "Gửi tin nhắn test thành công!" });
      } else {
        setTestResult({ success: false, msg: data.description || "Lỗi không xác định" });
      }
    } catch (err: any) {
      setTestResult({ success: false, msg: err.message });
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      {/* Header */}
      <div className="card">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-slate-500/10 flex items-center justify-center">
            <Settings2 className="w-5 h-5 text-slate-400" />
          </div>
          <div>
            <h2 className="font-bold text-white">Cấu hình hệ thống</h2>
            <p className="text-sm text-slate-400">
              Cài đặt thông báo và cấu hình chung
            </p>
          </div>
        </div>
      </div>

      {/* Telegram Config */}
      <div className="card space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
            <Bell className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">🤖 Thông báo Telegram</h3>
            <p className="text-sm text-slate-400">
              Cấu hình bot Telegram để nhận thông báo khi phát hiện phương tiện và vi phạm
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-slate-300 mb-1.5">
              Bot Token
            </label>
            <input
              type="password"
              value={botToken}
              onChange={(e) => setBotToken(e.target.value)}
              placeholder="Nhập Telegram Bot Token..."
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-slate-500 mt-1">
              Lấy từ @BotFather trên Telegram
            </p>
          </div>

          <div>
            <label className="block text-sm text-slate-300 mb-1.5">
              Chat ID
            </label>
            <input
              type="text"
              value={chatId}
              onChange={(e) => setChatId(e.target.value)}
              placeholder="Nhập Chat ID..."
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-slate-500 mt-1">
              Lấy từ @userinfobot hoặc @getidsbot
            </p>
          </div>

          <button
            onClick={handleTestTelegram}
            disabled={!botToken || !chatId}
            className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:text-slate-400 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
          >
            <Send className="w-4 h-4" />
            Gửi tin nhắn test
          </button>

          {testResult && (
            <div
              className={`p-3 rounded-lg flex items-start gap-2 ${
                testResult.success
                  ? "bg-green-500/10 border border-green-500/20"
                  : "bg-red-500/10 border border-red-500/20"
              }`}
            >
              {testResult.success ? (
                <CheckCircle2 className="w-4 h-4 text-green-400 shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              )}
              <p
                className={`text-sm ${
                  testResult.success ? "text-green-300" : "text-red-300"
                }`}
              >
                {testResult.msg}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Info */}
      <div className="card">
        <h3 className="font-semibold text-white mb-3">📝 Ghi chú</h3>
        <ul className="space-y-2 text-sm text-slate-300">
          <li>
            • Cấu hình Telegram được lưu trong biến môi trường{" "}
            <code className="px-1.5 py-0.5 bg-slate-700 rounded text-xs">
              TELEGRAM_BOT_TOKEN
            </code>{" "}
            và{" "}
            <code className="px-1.5 py-0.5 bg-slate-700 rounded text-xs">
              TELEGRAM_CHAT_ID
            </code>
          </li>
          <li>• Bạn có thể cập nhật token tại file backend/src/utils/notifications.py</li>
          <li>• Các ngưỡng phát hiện vi phạm được cấu hình trong tab Xử lý ảnh</li>
        </ul>
      </div>
    </div>
  );
}