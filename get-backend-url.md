# Get Your Backend URL

## 🎯 After Deploying Backend

Once you've deployed your backend to Render/Railway/etc., you'll get a URL like:

- **Render**: `https://vr-generation-backend.onrender.com`
- **Railway**: `https://vr-generation-production.up.railway.app`
- **Fly.io**: `https://your-app.fly.dev`

## 📋 Steps to Get Backend URL:

### For Render:
1. Go to your Render dashboard
2. Click on your service
3. Copy the URL from the top (e.g., `https://vr-generation-backend.onrender.com`)

### For Railway:
1. Go to your Railway dashboard
2. Click on your project
3. Copy the URL from the service overview

### For Fly.io:
1. Run `flyctl status` in your terminal
2. Copy the URL from the output

## 🔧 Set in Vercel:

1. Go to your Vercel project settings
2. Go to "Environment Variables"
3. Add: `NEXT_PUBLIC_API_URL` = `your-backend-url`
4. Redeploy your frontend

## ✅ Test Connection:

Visit: `https://your-backend-url/health`

You should see:
```json
{
  "status": "healthy",
  "message": "VR180 Backend is running",
  "timestamp": "2024-01-15 14:30:00",
  "uptime": "active",
  "keep_alive": "enabled"
}
```
