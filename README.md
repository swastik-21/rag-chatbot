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

3. Activate virtual environment:
```bash
source .venv/bin/activate
```

4. Start the chatbot:
```bash
./start_chatbot.sh
```

5. Open http://localhost:8080 in your browser

## API Usage

Streaming endpoint:
```bash
curl -X POST http://localhost:8080/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What products do you offer?"}'
```

Non-streaming endpoint:
```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What products do you offer?"}'
```

## Files

- `app.py` - FastAPI application with web interface
- `chatbot/` - Core RAG pipeline
- `docs/shopilots_site/` - Website data
- `start_chatbot.sh` - Startup script
- `scripts/scrape_shopilots.py` - Web scraper for updating data

## Rebuilding Index

To rebuild the vector database from documents:
```bash
python chatbot/memory_builder.py
```
