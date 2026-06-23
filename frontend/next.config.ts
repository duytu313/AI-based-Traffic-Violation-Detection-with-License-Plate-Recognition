import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    unoptimized: true,
  },
  // Allow large API responses
  experimental: {},
};

export default nextConfig;