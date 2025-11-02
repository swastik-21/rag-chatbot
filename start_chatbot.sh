#!/bin/bash

cd "$(dirname "$0")"
source .venv/bin/activate

export OPENAI_API_KEY="${OPENAI_API_KEY}"

echo "Starting Shopilots Chatbot..."
echo "Server will be available at http://localhost:8080"
echo ""
uvicorn app:app --host 0.0.0.0 --port 8080 --reload


