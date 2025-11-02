# Shopilots Chatbot

RAG chatbot for Shopilots website data using vector search and OpenAI API.

## Quick Start

### 1. Install Dependencies

```bash
poetry install
# OR
pip install -r requirements.txt
```

### 2. Build Vector Database

Before running the chatbot, you need to populate the vector database with documentation:

```bash
source .venv/bin/activate
export PYTHONPATH="${PWD}/chatbot:${PYTHONPATH}"
python3 chatbot/memory_builder.py
```

This loads documents from `docs/shopilots_site/` and creates the searchable index.

### 3. Run the Chatbot

```bash
# Optional: Set OpenAI API key (otherwise uses local model)
export OPENAI_API_KEY="your-api-key"

# Start the server
./start_chatbot.sh
```

Open http://localhost:8080 in your browser

## Rebuilding the Vector Database

If you update the documentation files, rebuild the index:

```bash
source .venv/bin/activate
export PYTHONPATH="${PWD}/chatbot:${PYTHONPATH}"
python3 chatbot/memory_builder.py --chunk-size 512 --chunk-overlap 25
```

## Deployment

Use the `Procfile` for Heroku/Railway/Render or `Dockerfile` for Docker deployments.
