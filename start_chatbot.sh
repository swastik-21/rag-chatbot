#!/bin/bash

cd "$(dirname "$0")"
source .venv/bin/activate

export OPENAI_API_KEY="${OPENAI_API_KEY}"

echo "Starting Shopilots Chatbot..."
python app.py


