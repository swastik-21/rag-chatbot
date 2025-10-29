# Quick Deploy Guide

## Option 1: Railway (Easiest - No CLI needed)

1. **Go to Railway**: https://railway.app
2. **Sign in with GitHub** (use your swastik-21 account)
3. **New Project** → **Deploy from GitHub repo**
4. **Select repository**: `swastik-21/rag-chatbot`
5. **Add Environment Variable**:
   - Click on your deployed service
   - Go to **Variables** tab
   - Add: `OPENAI_API_KEY` = your API key
6. **Deploy**: Railway auto-deploys on every push!

Your app will be live at: `https://your-app-name.railway.app`

## Option 2: Render (Direct Deployment)

### Step-by-Step Instructions:

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Sign in** with your GitHub account (swastik-21)
3. **Create New Web Service**:
   - Click "New +" button → Select "Web Service"
4. **Connect Repository**:
   - Authorize Render to access GitHub if prompted
   - Select repository: `swastik-21/rag-chatbot`
   - Click "Connect"
5. **Configure Service**:
   - **Name**: `rag-chatbot` (or any name you prefer)
   - **Region**: Choose closest to your users (e.g., US East)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. **Add Environment Variables**:
   - Click "Advanced" → "Add Environment Variable"
   - **Key**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key
   - Click "Save Changes"
7. **Deploy**:
   - Click "Create Web Service"
   - Wait for build and deployment (takes 5-10 minutes)

### Your App URL:
Once deployed, your app will be available at:
`https://your-service-name.onrender.com`

**Note**: The keep-alive mechanism is already built-in! Your service will stay awake automatically.

## Option 3: Heroku (Requires CLI)

If you want to use Heroku, install CLI first:

```bash
# macOS
brew tap heroku/brew && brew install heroku

# Then login and deploy
heroku login
heroku create your-app-name
heroku config:set OPENAI_API_KEY="your-key"
git push heroku main
```

## After Deployment

- Set `OPENAI_API_KEY` environment variable in your platform's dashboard
- The app will be accessible at the URL provided by your platform
- Make sure vector database is built (run `python chatbot/memory_builder.py` locally first)
