# Frontend Configuration Summary

## Files Created/Updated for Render Deployment

### Backend URL Configuration
- **script.js** - Updated to use production backend URL (`https://afdmi-123.onrender.com`) when not on localhost
- Environment-aware API endpoints configured in `script.js`:
  - Production backend: `https://afdmi-123.onrender.com/api`
  - Local development: `http://localhost:3001/api`

### Server & Deployment
- **server.js** - Express server to serve static files on Render
- **Dockerfile.frontend** - Docker configuration for containerized deployment
- **package.json** - Updated with Express dependency and start script
- **.dockerignore** - Optimizes Docker builds

### Configuration Files
- **render.yaml** - Render deployment configuration (auto-detected)
- **.env.example.frontend** - Environment variables template for frontend

### Documentation
- **RENDER_FRONTEND_DEPLOYMENT.md** - Complete deployment guide with troubleshooting

## Quick Start

### Option 1: Deploy on Render Dashboard (Recommended)

1. Push code to GitHub:
   ```bash
   git add .
   git commit -m "Configure frontend for Render deployment"
   git push origin main
   ```

2. Go to https://dashboard.render.com
3. Create new Web Service
4. Connect GitHub repository
5. Use these settings:
   - **Build Command:** `npm install`
   - **Start Command:** `npm start`
   - **Environment Variables:**
     - `NODE_ENV=production`
     - `BACKEND_URL=https://afdmi-123.onrender.com`

6. Click "Create Web Service"
7. Your frontend will deploy automatically

### Option 2: Deploy via Docker

```bash
docker build -t admi-frontend:latest -f Dockerfile.frontend .
docker run -p 3000:3000 -e BACKEND_URL=https://afdmi-123.onrender.com admi-frontend:latest
```

## Verification

After deployment, test:

1. **Frontend loads:** Visit your Render URL
2. **API works:** Open browser console and verify no CORS errors
3. **Backend connection:** Test an API call to verify backend responds

## Backend CORS Configuration

Ensure your backend has CORS enabled for your frontend URL:

```javascript
app.use(cors({
  origin: 'https://admi-frontend.onrender.com',
  credentials: true
}));
```

## Environment Variables

| Variable | Local | Production |
|----------|-------|-----------|
| `BACKEND_URL` | `http://localhost:3001` | `https://afdmi-123.onrender.com` |
| `NODE_ENV` | `development` | `production` |
| `PORT` | `3000` | Auto-assigned by Render |

The frontend automatically detects the environment and uses the correct backend URL.

## Support & Troubleshooting

See **RENDER_FRONTEND_DEPLOYMENT.md** for:
- Detailed setup steps
- Troubleshooting common issues
- Performance optimization tips
- Custom domain configuration
- CI/CD integration
