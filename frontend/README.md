# VR Platform Frontend

## Environment Variables

Create a `.env.local` file in the frontend directory with the following configuration:

```bash
# API Configuration
# For local development, use localhost
NEXT_PUBLIC_API_URL=http://localhost:8000

# For production deployment, replace with your deployed backend URL
# NEXT_PUBLIC_API_URL=https://your-backend-domain.com
```

## Development

```bash
npm install
npm run dev
```

## Production Deployment

1. Set the `NEXT_PUBLIC_API_URL` environment variable in your deployment platform (Vercel, Netlify, etc.)
2. Build and deploy the application

The frontend will automatically use the environment variable for all API calls to the backend.
