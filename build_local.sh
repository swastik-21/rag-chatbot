#!/bin/bash
# Local build script for Shopilots Chatbot

set -e

echo "🔨 Building Shopilots Chatbot locally..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run: python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check Python version
echo "📦 Python version: $(python3 --version)"
echo ""

# Install/update dependencies
echo "📥 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "⚠️  requirements.txt not found, skipping dependency installation"
fi
echo ""

# Build vector database
echo "🗄️  Building vector database..."
export PYTHONPATH="${PWD}/chatbot:${PYTHONPATH}"

if [ ! -d "docs/shopilots_site" ]; then
    echo "⚠️  Warning: docs/shopilots_site directory not found"
    echo "   Vector database will be empty"
else
    python3 chatbot/memory_builder.py --chunk-size 512 --chunk-overlap 25
    echo "✓ Vector database built"
fi
echo ""

# Verify build
echo "✅ Verifying build..."
python3 -c "
import sys
sys.path.insert(0, '.')
from app import initialize_components, index

result = initialize_components()
if not result:
    print('❌ Initialization failed')
    sys.exit(1)

if index is None:
    print('❌ Vector database not initialized')
    sys.exit(1)

results = index.collection.get(limit=1)
doc_count = len(results.get('ids', []))
print(f'✓ Vector database has {doc_count} documents')
" 2>&1 | grep -v "llama\|ggml\|Metal\|token" || true

echo ""
echo "🎉 Build complete!"
echo ""
echo "To start the server:"
echo "  source .venv/bin/activate"
echo "  uvicorn app:app --host 0.0.0.0 --port 8080 --reload"
echo ""
echo "Or use: ./start_chatbot.sh"
echo ""

