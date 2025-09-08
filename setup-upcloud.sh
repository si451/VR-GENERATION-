#!/bin/bash

echo "🔧 Setting up UpCloud server for VR180 Backend..."

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
echo "📦 Installing Docker Compose..."
apt install docker-compose -y

# Add user to docker group
echo "👤 Adding user to docker group..."
usermod -aG docker root

# Create workspace directory
echo "📁 Creating workspace directory..."
mkdir -p /app/workspace
chmod 755 /app/workspace

# Install Git (if not already installed)
echo "📥 Installing Git..."
apt install git -y

echo "✅ UpCloud server setup complete!"
echo "🚀 You can now deploy your backend using the deploy-upcloud.sh script"
