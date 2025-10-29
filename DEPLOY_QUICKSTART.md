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

## Option 2: Render (Alternative)

1. Go to https://render.com
2. Sign in with GitHub
3. New → Web Service
4. Connect `swastik-21/rag-chatbot` repository
5. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Add Environment Variable: `OPENAI_API_KEY`
7. Deploy!

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
