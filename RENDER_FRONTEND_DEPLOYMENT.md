# Frontend Deployment on Render

This guide covers deploying the ADMI frontend to Render as a static website served by Node.js.

## Prerequisites

- Render account (free or paid)
- GitHub repository with frontend code
- Backend already deployed at `https://afdmi-123.onrender.com`

## Deployment Steps

### Option 1: Deploy via Render Dashboard (Recommended)

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Configure frontend for Render deployment"
   git push origin main
   ```

2. **Create new Web Service on Render**
   - Go to https://dashboard.render.com
   - Click "New +"
   - Select "Web Service"
   - Connect your GitHub repository
   - Select the branch (e.g., main)

3. **Configure Service**
   - **Name:** `admi-frontend`
   - **Runtime:** Node
   - **Build Command:** `npm install`
   - **Start Command:** `npm start`
   - **Plan:** Free or paid (upgrade for production)

4. **Set Environment Variables**
   In the "Environment" section, add:
   - `NODE_ENV` = `production`
   - `BACKEND_URL` = `https://afdmi-123.onrender.com`
   - `PORT` = (leave empty - Render assigns automatically)

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy
   - Your frontend will be available at `https://admi-frontend.onrender.com` (or your custom domain)

### Option 2: Deploy via Render YAML (render.yaml)

If your repository root has a `render.yaml` file, Render auto-detects it:

```bash
git push origin main
```

Render will automatically read `render.yaml` and deploy with the specified configuration.

### Option 3: Deploy via Docker

1. **Build locally (optional testing)**
   ```bash
   docker build -t admi-frontend:latest -f Dockerfile.frontend .
   docker run -d -p 3000:3000 -e BACKEND_URL=https://afdmi-123.onrender.com admi-frontend:latest
   ```

2. **Push to Docker registry** (Docker Hub, GitHub Container Registry, etc.)
   ```bash
   docker tag admi-frontend:latest yourusername/admi-frontend:latest
   docker push yourusername/admi-frontend:latest
   ```

3. **Deploy on Render**
   - Create Web Service from Docker image
   - Enter image URL: `yourusername/admi-frontend:latest`
   - Set environment variables as above

## Verification

Once deployed, test the frontend:

1. **Check Health**
   ```bash
   curl https://admi-frontend.onrender.com
   ```

2. **Open in Browser**
   - Navigate to your frontend URL
   - Check browser console for errors
   - Test API calls to backend

3. **Monitor Logs**
   - Go to Render dashboard
   - Click your service
   - View "Logs" tab for real-time output

## Troubleshooting

### Frontend won't load / 502 Bad Gateway
- Check service logs on Render dashboard
- Verify `npm start` command works locally
- Ensure `server.js` exists and is correct

### API calls to backend failing
- Verify backend URL is correct: `https://afdmi-123.onrender.com`
- Check backend is running: `curl https://afdmi-123.onrender.com`
- Review browser console for CORS errors
- Ensure backend has CORS enabled for frontend URL

### Build failing
- Check "Build Logs" tab in Render
- Ensure `package.json` is valid JSON
- Verify all dependencies are correct
- Try building locally first: `npm install && npm start`

### Static files not loading (CSS, JS images)
- Check that `server.js` serves static files correctly
- Verify `express.static()` path is correct
- Test locally first

## Custom Domain

To use a custom domain:

1. In Render dashboard, go to your service settings
2. Under "Custom Domain," enter your domain
3. Update DNS records with Render's provided values
4. Wait for SSL certificate (auto-issued)

## Environment Variables Reference

| Variable | Value | Purpose |
|----------|-------|---------|
| `NODE_ENV` | `production` | Disables debug logging |
| `BACKEND_URL` | `https://afdmi-123.onrender.com` | Backend API endpoint |
| `PORT` | (auto-assigned by Render) | Server port |

## Performance Tips

- Use `npm install --production` to exclude dev dependencies
- Compress large assets (CSS, JS, images)
- Enable gzip compression in Express
- Use CDN for static assets if needed

## Updating the Frontend

To push updates:

```bash
# Make changes locally
git add .
git commit -m "Update frontend"
git push origin main
```

Render will automatically rebuild and redeploy.

## Backend Configuration

Ensure your backend (`https://afdmi-123.onrender.com`) is configured for CORS:

```javascript
// In your backend (main.js or similar)
app.use(cors({
  origin: [
    'https://admi-frontend.onrender.com',
    'http://localhost:3000' // for local development
  ],
  credentials: true
}));
```

## Support

For issues:
1. Check Render documentation: https://render.com/docs
2. Review service logs on Render dashboard
3. Test backend connectivity
4. Check frontend console (F12 → Console tab)
