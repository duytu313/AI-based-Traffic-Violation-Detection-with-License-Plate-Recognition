import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Traffic AI - Hệ thống quản lý giao thông thông minh",
  description:
    "Hệ thống phát hiện vi phạm giao thông và nhận dạng biển số xe sử dụng AI",
  icons: {
    icon: "/image/logo.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}