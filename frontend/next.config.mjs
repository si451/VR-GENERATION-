/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // Reduce hot reloading frequency to prevent constant refreshes
  webpack: (config, { dev, isServer }) => {
    if (dev && !isServer) {
      config.watchOptions = {
        poll: 1000, // Check for changes every 1 second instead of default
        aggregateTimeout: 300, // Wait 300ms before rebuilding
        ignored: /node_modules/, // Ignore node_modules changes
      }
    }
    return config
  },
}

export default nextConfig
