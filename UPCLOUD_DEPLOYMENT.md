# 🚀 UpCloud Deployment Guide

## Prerequisites
- UpCloud account with free trial
- SSH key pair (generate if you don't have one)

## Step 1: Create UpCloud Server

1. **Log into UpCloud dashboard**
2. **Go to "Servers"** → **"Deploy server"**
3. **Configure server:**
   - **Zone**: Choose closest to your location
   - **Operating System**: Ubuntu 22.04 LTS
   - **Plan**: 2GB RAM, 1 vCPU (free trial)
   - **Storage**: 20GB SSD
   - **Networking**: Public IPv4
   - **SSH Keys**: Add your public SSH key

4. **Deploy server** and wait for it to be ready

## Step 2: Connect to Server

```bash
ssh root@YOUR_SERVER_IP
```

## Step 3: Setup Server

Upload and run the setup script:

```bash
# Upload setup script to server
scp setup-upcloud.sh root@YOUR_SERVER_IP:/root/

# Connect to server
ssh root@YOUR_SERVER_IP

# Make script executable and run
chmod +x setup-upcloud.sh
./setup-upcloud.sh
```

## Step 4: Upload Your Code

```bash
# From your local machine, upload the backend code
scp -r backend/ root@YOUR_SERVER_IP:/root/
scp deploy-upcloud.sh root@YOUR_SERVER_IP:/root/
```

## Step 5: Deploy Backend

```bash
# On the server
chmod +x deploy-upcloud.sh
./deploy-upcloud.sh
```

## Step 6: Test Deployment

```bash
# Check if container is running
docker ps

# Check logs
docker logs vr180-backend-container

# Test API
curl http://localhost:8000/health
```

## Step 7: Configure Firewall

```bash
# Allow port 8000
ufw allow 8000
ufw enable
```

## Step 8: Update Frontend

Update your frontend's API URL to point to your UpCloud server:

```bash
# In your frontend .env file
NEXT_PUBLIC_API_URL=http://YOUR_SERVER_IP:8000
```

## Monitoring

```bash
# Check container status
docker ps

# View logs
docker logs vr180-backend-container

# Restart container
docker restart vr180-backend-container

# Update deployment
./deploy-upcloud.sh
```

## Troubleshooting

### Container won't start:
```bash
docker logs vr180-backend-container
```

### Out of memory:
```bash
# Check memory usage
free -h
docker stats
```

### Port issues:
```bash
# Check if port is open
netstat -tlnp | grep 8000
```

## Environment Variables

The following environment variables are set automatically:
- `PORT=8000`
- `WORKSPACE_DIR=/app/workspace`

## File Structure on Server

```
/root/
├── backend/
│   └── api/
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── main.py
│       └── ...
├── setup-upcloud.sh
├── deploy-upcloud.sh
└── /app/workspace/ (for video processing)
```

## Next Steps

1. Test your video processing pipeline
2. Monitor memory usage during processing
3. Set up monitoring and logging
4. Configure domain name (optional)
5. Set up SSL certificate (optional)
