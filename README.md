# Shopilots Chatbot

RAG chatbot for Shopilots website data using vector search and OpenAI API.

## Quick Start

1. Set OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

2. Install dependencies:
```bash
poetry install
```

3. Start the chatbot:
```bash
./start_chatbot.sh
```

4. Open http://localhost:8080 in your browser

## Deployment

### Render

1. Go to https://dashboard.render.com
2. New + → Web Service
3. Connect repository: `swastik-21/rag-chatbot`
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `OPENAI_API_KEY`
7. Deploy

## Files

- `app.py` - FastAPI application
- `chatbot/` - RAG pipeline
- `docs/shopilots_site/` - Website data
