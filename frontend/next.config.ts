import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    unoptimized: true,
  },
  // Allow large API responses
  experimental: {},
  // Proxy API requests to backend
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
      {
        source: "/data/:path*",
        destination: "http://localhost:8000/data/:path*",
      },
    ];
  },
};

export default nextConfig;