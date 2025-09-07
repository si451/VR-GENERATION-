# Frontend Deployment Guide

## 🚀 Deploy to Vercel

### Step 1: Connect to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Sign up/Login with GitHub
3. Click "New Project"
4. Import your repository: `si451/VR-GENERATION-`
5. Select the `frontend` folder as the root directory

### Step 2: Configure Environment Variables

In Vercel dashboard, go to your project settings and add:

```
NEXT_PUBLIC_API_URL=https://vr-generation-production.up.railway.app
```

**This connects your frontend to the Railway backend deployment.**

### Step 3: Build Settings

- **Framework Preset**: Next.js
- **Root Directory**: `frontend`
- **Build Command**: `npm run build`
- **Output Directory**: `.next`

### Step 4: Deploy

Click "Deploy" and wait for the build to complete.

## 🔧 Environment Variables

### Required:
- `NEXT_PUBLIC_API_URL` - Your backend API URL

### Optional:
- `NEXT_PUBLIC_APP_NAME` - App name for display
- `NEXT_PUBLIC_APP_VERSION` - App version

## 📝 Notes

- The `NEXT_PUBLIC_` prefix is required for client-side access
- Environment variables are available at build time
- Changes to env vars require a new deployment
