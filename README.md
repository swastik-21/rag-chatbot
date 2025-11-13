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

## Analytics

The chatbot includes comprehensive analytics tracking:

### Available Endpoints

- `GET /api/analytics/kpis` - Key Performance Indicators (completion rate, response time, etc.)
- `GET /api/analytics/top-questions?limit=10` - Most common questions
- `GET /api/analytics/hourly-stats?hours=24` - Hourly question statistics
- `GET /api/analytics/model-usage` - Model usage statistics
- `GET /api/analytics/product-categories` - Product category statistics
- `GET /api/analytics/fallback-reasons` - Fallback reason statistics
- `GET /api/analytics/recent-events?limit=100` - Recent conversation events
- `GET /api/analytics/session/{session_id}` - Session-specific statistics
- `GET /api/analytics/export?format=json` - Export analytics (json or csv)

### Tracked Metrics

- **Conversation Metrics**: Total conversations, questions, answers
- **Performance KPIs**: Completion rate, fallback rate, error rate, response time
- **User Engagement**: Questions per session, first contact resolution rate
- **Model Analytics**: Which model (OpenAI vs Llama) is being used
- **Product Categories**: Tracks which product categories are discussed most
- **Fallback Tracking**: When and why the bot falls back to simpler responses
- **Error Tracking**: Error types and frequencies

### Product Categories

The chatbot recognizes and tracks the following product categories:

**AI Sales Agents:**
- Website Agent
- Social Media Agent
- Messenger Agent
- Call Agent
- GPT Store

**Industry Solutions:**
- Electronics & Tech (+53% conversion rate, +41% AOV)
- Fashion & Apparel (+49% conversion rate, +32% AOV)
- Home & Garden (+47% conversion rate, +38% AOV)
- Agencies & Partners

**Platforms & Integrations:**
- E-commerce: Shopify, WooCommerce, Magento
- Social Media: Facebook, Messenger, Telegram, WhatsApp
- AI Platforms: ChatGPT, GPT Store

## Deployment

Use the `Procfile` for Heroku/Railway/Render or `Dockerfile` for Docker deployments.
