# Deploy to Render - Quick Guide

## Repository
**GitHub**: https://github.com/swastik-21/rag-chatbot

## Quick Deploy Steps

1. **Visit Render Dashboard**
   - Go to: https://dashboard.render.com
   - Sign in with GitHub (swastik-21 account)

2. **Create Web Service**
   - Click "New +" → "Web Service"
   - Connect repository: `swastik-21/rag-chatbot`

3. **Configuration**
   ```
   Name: rag-chatbot
   Region: US East (or closest)
   Branch: main
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

4. **Environment Variables**
   - Key: `OPENAI_API_KEY`
   - Value: (your OpenAI API key)

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

## Features Already Included

✅ Keep-alive mechanism (prevents sleep)  
✅ Health check endpoint (`/health`)  
✅ Streaming API (`/api/chat/stream`)  
✅ Web interface at root (`/`)  

## After Deployment

Your app will be live at:
`https://your-service-name.onrender.com`

The service will automatically stay awake thanks to the built-in keep-alive mechanism!

## Troubleshooting

- **Build fails**: Check logs in Render dashboard
- **Service sleeps**: Keep-alive should prevent this (pings every 5 min)
- **Environment variables**: Ensure `OPENAI_API_KEY` is set correctly

