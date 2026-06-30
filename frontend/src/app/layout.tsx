import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Traffic AI - Smart Traffic Management System",
  description:
    "AI-powered traffic violation detection and license plate recognition system",
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
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}