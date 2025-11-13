from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
import traceback
import json
from typing import Optional, List
import asyncio
import httpx
import time
import gc
import os

# Optimize Python memory settings for cloud deployments
if os.getenv("RENDER") or os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RENDER_EXTERNAL_URL"):
    # Force garbage collection before loading heavy models
    gc.collect()
    # Set environment variables to reduce memory usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism to save memory

sys.path.insert(0, str(Path(__file__).parent / "chatbot"))

from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
from bot.conversation.conversation_handler import answer_with_context, refine_question
from bot.conversation.ctx_strategy import get_ctx_synthesis_strategy
from bot.model.model_registry import Model, get_model_settings
from helpers.analytics import analytics_tracker, ConversationEvent
import os
from openai import OpenAI
from datetime import datetime

app = FastAPI(title="Shopilots Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = None
index = None
synthesis_strategy = None
chat_histories = {}
components_loaded = False
openai_client = None
use_openai = False

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None

def initialize_components():
    global llm, index, synthesis_strategy, components_loaded, openai_client, use_openai
    
    root_folder = Path(__file__).resolve().parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"
    
    # Check if we're on Render or Railway (cloud deployment)
    is_cloud = os.getenv("RENDER") or os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RENDER_EXTERNAL_URL")
    
    try:
        # Check if vector store path exists
        if not vector_store_path.exists():
            print(f"WARNING: Vector store path does not exist: {vector_store_path}")
            print("Creating directory...")
            vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB first (lighter)
        # We'll load the embedding model lazily if needed, but for now load it
        # since we need it for queries
        print("Initializing embedding model...")
        embedding = Embedder()
        print("Embedding model loaded")
        
        # Force garbage collection after loading embedding model
        if is_cloud:
            gc.collect()
        
        print(f"Initializing vector database at: {vector_store_path}")
        # Initialize ChromaDB with memory optimizations
        index = Chroma(persist_directory=str(vector_store_path), embedding=embedding)
        print("Vector database initialized")
        
        # Verify index is working
        try:
            test_results = index.collection.get(limit=1)
            print(f"Vector database verified: {len(test_results.get('ids', []))} documents found")
        except Exception as e:
            print(f"WARNING: Vector database verification failed: {e}")
            print("Database may be empty, but continuing...")
        
        # Force garbage collection after loading vector database
        if is_cloud:
            gc.collect()
        
        # Always prioritize OpenAI on cloud deployments to save memory
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                openai_client = OpenAI(api_key=api_key)
                use_openai = True
                components_loaded = True
                print("Using OpenAI API")
                return True
            except Exception as e:
                print(f"OpenAI init failed: {e}")
                if is_cloud:
                    print("On cloud deployment - OpenAI is required. Please set OPENAI_API_KEY.")
                    components_loaded = False
                    return False
        
        # Only load local model if not on cloud and OpenAI is not available
        if is_cloud:
            print("On cloud deployment - local model not supported due to memory constraints.")
            print("Please set OPENAI_API_KEY environment variable.")
            components_loaded = False
            return False
        
        # Local development: try to load local model
        try:
            model_name = Model.LLAMA_3_2_three.value
            model_settings = get_model_settings(model_name)
            model_folder.mkdir(parents=True, exist_ok=True)
            llm = LamaCppClient(model_folder=model_folder, model_settings=model_settings)
            synthesis_strategy = get_ctx_synthesis_strategy("create-and-refine", llm=llm)
            components_loaded = True
            use_openai = False
            print("Using local LLM (Llama 3.2 3B)")
        except Exception as e:
            print(f"Local model init failed: {e}")
            components_loaded = False
        
        return True
    except Exception as e:
        print(f"Error initializing components: {e}")
        import traceback
        traceback.print_exc()
        # Ensure index is None on failure
        index = None
        components_loaded = False
        return False

def get_chat_history(session_id: str) -> ChatHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatHistory(total_length=6)
    return chat_histories[session_id]

def detect_product_category(question: str, answer: str = "") -> Optional[str]:
    """Detect product category from question or answer"""
    text = (question + " " + answer).lower()
    
    # AI Sales Agents
    if "website agent" in text:
        return "Website Agent"
    if "social media agent" in text:
        return "Social Media Agent"
    if "messenger agent" in text:
        return "Messenger Agent"
    if "call agent" in text:
        return "Call Agent"
    if "gpt store" in text or "chatgpt" in text:
        return "GPT Store"
    
    # Industry Solutions
    if "electronics" in text or ("tech" in text and "agent" not in text) or "gaming" in text:
        return "Electronics & Tech"
    if "fashion" in text or "apparel" in text or "clothing" in text:
        return "Fashion & Apparel"
    if "home" in text or "garden" in text or "bedding" in text or "decor" in text:
        return "Home & Garden"
    if "agency" in text or ("partner" in text and "agent" not in text) or "white-label" in text:
        return "Agencies & Partners"
    
    return None

def retrieve_docs(query: str, k: int = 5):
    if index is None:
        return [], []
    
    results = []
    sources_list = []
    
    # Always try similarity_search first (most reliable)
    try:
        docs = index.similarity_search(query, k=k*2)  # Get more to filter better
        for doc in docs:
            results.append(doc)
            sources_list.append({
                "document": doc.metadata.get("source", "Unknown"),
                "score": 1.0,
            })
    except Exception as e:
        print(f"Similarity search error: {e}")
    
    # Try threshold search as supplement (lower threshold)
    try:
        docs2, sources2 = index.similarity_search_with_threshold(query, k=k*2, threshold=0.01)
        seen_content = {doc.page_content[:100] for doc in results}
        for doc, source in zip(docs2, sources2):
            if doc.page_content[:100] not in seen_content and len(doc.page_content.strip()) > 20:
                results.append(doc)
                sources_list.append(source)
                seen_content.add(doc.page_content[:100])
    except Exception as e:
        print(f"Threshold search error: {e}")
    
    unique_results = []
    unique_sources = []
    seen = set()
    for doc, source in zip(results, sources_list):
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen and len(doc.page_content.strip()) > 15:
            unique_results.append(doc)
            unique_sources.append(source)
            seen.add(content_hash)
    
    # If we still have no results, do a broad search
    if not unique_results:
        try:
            fallback_docs = index.similarity_search("shopilots products agents", k=k)
            for doc in fallback_docs:
                if len(doc.page_content.strip()) > 15:
                    unique_results.append(doc)
                    unique_sources.append({
                        "document": doc.metadata.get("source", "Unknown"),
                        "score": 0.5,
                    })
        except:
            pass
    
    return unique_results[:k], unique_sources[:k]

def format_response(docs: List, query: str) -> str:
    if not docs:
        return "I couldn't find specific information about that. Could you try rephrasing your question?"
    
    query_lower = query.lower()
    all_text = " ".join([doc.page_content for doc in docs[:3]]).lower()
    
    # Extract AI Sales Agents (Products)
    if any(word in query_lower for word in ["product", "offer", "provide", "what do you", "services", "agent", "sales agent"]):
        products = []
        
        # AI Sales Agents
        if "website agent" in all_text:
            products.append("• Website Agent - Embed on your e-commerce site. Deploy conversational salesforce that sell — not just chat.")
        if "social media agent" in all_text:
            products.append("• Social Media Agent - Engage on social platforms with AI-powered shopping assistance.")
        if "messenger agent" in all_text:
            products.append("• Messenger Agent - Chat on WhatsApp & Messenger. Handle customer inquiries and drive sales through messaging platforms.")
        if "call agent" in all_text:
            products.append("• Call Agent - Handle customer phone calls with AI voice capabilities.")
        if "gpt store" in all_text or "chatgpt" in all_text:
            products.append("• GPT Store - Launch in ChatGPT & GPT Store. Create a custom GPT shopping assistant.")
        
        # Industry Solutions
        if "electronics" in all_text or "tech" in all_text or "gaming" in all_text:
            products.append("• Electronics & Tech - Help customers navigate complex tech specs and find the perfect device. Results: +53% conversion rate, +41% AOV")
        if "fashion" in all_text or "apparel" in all_text or "clothing" in all_text:
            products.append("• Fashion & Apparel - Provide personalized styling advice and size recommendations. Results: +49% conversion rate, +32% AOV")
        if "home" in all_text or "garden" in all_text or "bedding" in all_text or "decor" in all_text:
            products.append("• Home & Garden - Guide customers through home improvement and decor decisions. Results: +47% conversion rate, +38% AOV")
        if "agency" in all_text or "partner" in all_text or "white-label" in all_text:
            products.append("• Agencies & Partners - White-label solutions for agencies serving multiple retail clients.")
        
        if products:
            return "Shopilots offers the following products and solutions:\n\n" + "\n".join(products)
    
    # Extract Industry Solutions specifically
    elif any(word in query_lower for word in ["industry", "solution", "vertical", "sector", "category"]):
        solutions = []
        
        if "electronics" in all_text or "tech" in all_text or "gaming" in all_text:
            solutions.append("• Electronics & Tech - Smart product comparisons, technical specification matching, compatibility checking, performance recommendations. Results: +53% conversion rate, +41% AOV")
        if "fashion" in all_text or "apparel" in all_text or "clothing" in all_text:
            solutions.append("• Fashion & Apparel - Style matching, size and fit guidance, seasonal trend advice, outfit coordination. Results: +49% conversion rate, +32% AOV")
        if "home" in all_text or "garden" in all_text or "bedding" in all_text or "decor" in all_text:
            solutions.append("• Home & Garden - Room design consultation, appliance recommendations, space optimization, maintenance guidance. Results: +47% conversion rate, +38% AOV")
        if "agency" in all_text or "partner" in all_text or "white-label" in all_text:
            solutions.append("• Agencies & Partners - Multi-client management, custom branding, analytics and reporting, agency partner support")
        
        if solutions:
            return "Shopilots provides industry-specific solutions:\n\n" + "\n".join(solutions)
    
    # Extract platforms
    elif any(word in query_lower for word in ["platform", "integrate", "shopify", "woocommerce", "supported", "integration"]):
        platforms = []
        
        # E-commerce Platforms
        if "shopify" in all_text:
            platforms.append("• Shopify")
        if "woocommerce" in all_text:
            platforms.append("• WooCommerce")
        if "magento" in all_text:
            platforms.append("• Magento")
        
        # Social Media & Messaging
        if "whatsapp" in all_text or "messenger" in all_text:
            platforms.append("• WhatsApp & Messenger")
        if "telegram" in all_text:
            platforms.append("• Telegram")
        if "facebook" in all_text:
            platforms.append("• Facebook")
        
        # AI Platforms
        if "chatgpt" in all_text or "gpt store" in all_text:
            platforms.append("• ChatGPT & GPT Store")
        
        if platforms:
            return "Shopilots integrates with:\n\n" + "\n".join(platforms)
    
    # Extract pricing
    elif any(word in query_lower for word in ["price", "pricing", "plan", "cost", "subscription"]):
        return "Shopilots offers:\n\n• Performance Plan - Free to start (up to 500 conversations/month)\n• Business Plan - $299/month (up to 5,000 conversations/month)\n• Enterprise Plan - Custom pricing (unlimited conversations, dedicated support)"
    
    # Extract features
    elif any(word in query_lower for word in ["feature", "capability", "advantage", "benefit"]):
        features = []
        
        if "catalog" in all_text or "integration" in all_text:
            features.append("• Real-time catalog integration")
        if "sales" in all_text or "conversion" in all_text:
            features.append("• Sales-focused conversations")
        if "channel" in all_text or "multi-channel" in all_text:
            features.append("• Multi-channel deployment")
        if "analytics" in all_text:
            features.append("• Revenue-driven analytics")
        if "voice" in all_text or "brand" in all_text:
            features.append("• Brand voice customization")
        if "checkout" in all_text:
            features.append("• Fast checkout integration")
        if "booking" in all_text or "appointment" in all_text:
            features.append("• Booking appointment feature")
        
        if features:
            return "Shopilots key features:\n\n" + "\n".join(features)
    
    # Default: clean and return relevant content
    content = docs[0].page_content.strip()
    if content.startswith("Source: http"):
        lines = content.split("\n", 1)
        content = lines[1].strip() if len(lines) > 1 else content
    
    return content[:400] + "..." if len(content) > 400 else content

async def keep_alive_ping():
    """Keep service alive by pinging health endpoint every 5 minutes"""
    await asyncio.sleep(60)  # Initial delay
    
    port = os.getenv("PORT", "8080")
    # Try external URL first (for Render/Railway), fallback to localhost
    external_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("RAILWAY_PUBLIC_DOMAIN")
    local_url = f"http://127.0.0.1:{port}"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes between pings
                # Try external URL if available, otherwise use localhost
                if external_url:
                    try:
                        await client.get(f"{external_url}/health")
                        print(f"Keep-alive ping successful to {external_url}/health")
                    except Exception as e:
                        # Fallback to localhost if external ping fails
                        print(f"External ping failed: {e}, trying localhost")
                        await client.get(f"{local_url}/health")
                else:
                    await client.get(f"{local_url}/health")
                    print(f"Keep-alive ping successful to {local_url}/health")
            except Exception as e:
                print(f"Keep-alive ping failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint to keep service alive"""
    return {"status": "ok", "service": "shopilots-chatbot"}

@app.get("/api/analytics/kpis")
async def get_kpis():
    """Get key performance indicators"""
    return analytics_tracker.get_kpis()

@app.get("/api/analytics/top-questions")
async def get_top_questions(limit: int = 10):
    """Get most common questions"""
    return {"questions": analytics_tracker.get_top_questions(limit)}

@app.get("/api/analytics/hourly-stats")
async def get_hourly_stats(hours: int = 24):
    """Get hourly statistics"""
    return {"stats": analytics_tracker.get_hourly_stats(hours)}

@app.get("/api/analytics/model-usage")
async def get_model_usage():
    """Get model usage statistics"""
    return analytics_tracker.get_model_usage_stats()

@app.get("/api/analytics/product-categories")
async def get_product_categories():
    """Get product category statistics"""
    return analytics_tracker.get_product_category_stats()

@app.get("/api/analytics/fallback-reasons")
async def get_fallback_reasons():
    """Get fallback reason statistics"""
    return analytics_tracker.get_fallback_reasons()

@app.get("/api/analytics/recent-events")
async def get_recent_events(limit: int = 100):
    """Get recent conversation events"""
    return {"events": analytics_tracker.get_recent_events(limit)}

@app.get("/api/analytics/session/{session_id}")
async def get_session_stats(session_id: str):
    """Get statistics for a specific session"""
    stats = analytics_tracker.get_session_stats(session_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return stats

@app.get("/api/analytics/export")
async def export_analytics(format: str = "json"):
    """Export analytics data"""
    if format not in ["json", "csv"]:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
    
    data = analytics_tracker.export_events(format)
    if format == "json":
        return {"data": json.loads(data)}
    else:
        from fastapi.responses import Response
        return Response(content=data, media_type="text/csv", headers={
            "Content-Disposition": "attachment; filename=analytics.csv"
        })

@app.on_event("startup")
async def startup_event():
    global components_loaded
    initialize_components()
    asyncio.create_task(keep_alive_ping())

@app.get("/")
async def read_root():
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shopilots Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #f5f5f5; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background: #667eea; color: white; padding: 20px; border-radius: 8px 8px 0 0; }
        .header h1 { font-size: 24px; }
        .chat-area { height: 500px; overflow-y: auto; padding: 20px; }
        .message { margin-bottom: 15px; }
        .message.user { text-align: right; }
        .message-content { display: inline-block; padding: 12px 16px; border-radius: 12px; max-width: 70%; }
        .message.assistant .message-content { background: #f0f0f0; }
        .message.user .message-content { background: #667eea; color: white; }
        .input-area { padding: 20px; border-top: 1px solid #e0e0e0; display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 6px; }
        .input-area button { padding: 12px 24px; background: #667eea; color: white; border: none; border-radius: 6px; cursor: pointer; }
        .input-area button:disabled { opacity: 0.5; }
        .typing-indicator { display: inline-block; padding: 12px 16px; }
        .typing-indicator span { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #666; margin: 0 2px; animation: typing 1.4s infinite; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing { 0%, 60%, 100% { transform: translateY(0); opacity: 0.7; } 30% { transform: translateY(-10px); opacity: 1; } }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0.3; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Shopilots Chatbot</h1>
        </div>
        <div class="chat-area" id="chatArea">
            <div class="message assistant">
                <div class="message-content">Hello! I'm Shopilots Assistant. I can help you with questions about Shopilots products, platforms, and services.</div>
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Ask a question..." />
            <button id="sendBtn">Send</button>
        </div>
    </div>
    <script>
        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function sendMessage() {
            const sendBtn = document.getElementById('sendBtn');
            const userInput = document.getElementById('userInput');
            const chatArea = document.getElementById('chatArea');
            
            if (!sendBtn || !userInput || !chatArea) {
                return;
            }
            
            const question = userInput.value.trim();
            if (!question || sendBtn.disabled) return;
            
            chatArea.innerHTML += '<div class="message user"><div class="message-content">' + escapeHtml(question) + '</div></div>';
            
            // Add typing indicator
            const typingId = 'typing-' + Date.now();
            chatArea.innerHTML += '<div class="message assistant" id="' + typingId + '"><div class="message-content typing-indicator"><span></span><span></span><span></span></div></div>';
            chatArea.scrollTop = chatArea.scrollHeight;
            
            userInput.value = '';
            sendBtn.disabled = true;
            
            // Use streaming endpoint
            fetch('/api/chat/stream', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ question: question }),
                cache: 'no-cache'
            })
            .then(async response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let fullText = '';
                
                // Replace typing indicator with actual message
                const messageDiv = document.getElementById(typingId);
                messageDiv.innerHTML = '<div class="message-content"></div>';
                const contentDiv = messageDiv.querySelector('.message-content');
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    const newline = String.fromCharCode(10);
                    const lines = buffer.split(newline);
                    buffer = lines.pop(); // Keep incomplete line in buffer
                    
                    for (const line of lines) {
                        if (!line.trim()) continue;
                        
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6).trim();
                            if (data === '[DONE]') {
                                continue;
                            }
                            if (!data) continue;
                            
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.token) {
                                    fullText += parsed.token;
                                    // Update immediately for real-time effect - no requestAnimationFrame delay
                                    contentDiv.innerHTML = escapeHtml(fullText) + '<span style="opacity: 0.6; animation: blink 1s infinite">▋</span>';
                                    chatArea.scrollTop = chatArea.scrollHeight;
                                } else if (parsed.error) {
                                    throw new Error(parsed.error);
                                }
                            } catch (e) {
                                // Not JSON, might be plain token - append directly
                                if (data && data !== '[DONE]') {
                                    fullText += data;
                                    contentDiv.innerHTML = escapeHtml(fullText) + '<span style="opacity: 0.6; animation: blink 1s infinite">▋</span>';
                                    chatArea.scrollTop = chatArea.scrollHeight;
                                }
                            }
                        }
                    }
                }
                
                // Remove cursor
                contentDiv.innerHTML = escapeHtml(fullText);
                chatArea.scrollTop = chatArea.scrollHeight;
                const sendBtnEl = document.getElementById('sendBtn');
                if (sendBtnEl) sendBtnEl.disabled = false;
            })
            .catch(err => {
                const errorDiv = document.getElementById(typingId);
                if (errorDiv) {
                    errorDiv.innerHTML = '<div class="message-content">Error: ' + escapeHtml(err.message) + '</div>';
                }
                const sendBtnEl = document.getElementById('sendBtn');
                if (sendBtnEl) sendBtnEl.disabled = false;
            });
        }
        
        // Initialize when page loads
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initChat);
        } else {
            initChat();
        }
        
        function initChat() {
            const sendBtn = document.getElementById('sendBtn');
            const userInput = document.getElementById('userInput');
            
            if (!sendBtn || !userInput) {
                setTimeout(initChat, 100);
                return;
            }
            
            sendBtn.onclick = null;
            userInput.onkeypress = null;
            
            sendBtn.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                sendMessage();
                return false;
            };
            
            userInput.onkeypress = function(e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    e.stopPropagation();
                    const btn = document.getElementById('sendBtn');
                    if (btn && !btn.disabled) {
                        sendMessage();
                    }
                    return false;
                }
            };
        }
    </script>
</body>
</html>"""
    return HTMLResponse(html)

async def generate_streaming_response(question: str, docs: list):
    """Generate streaming response using OpenAI or local LLM"""
    if not components_loaded or not docs:
        answer = format_response(docs, question) if docs else "I couldn't find specific information about that."
        for i in range(0, len(answer), 5):
            chunk = answer[i:i+5]
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    context_text = "\n\n".join([doc.page_content for doc in docs[:3]])
    
    system_prompt = """You are Shopilots Assistant, a helpful AI assistant for Shopilots, an AI-powered e-commerce sales platform. When introducing yourself, say "I am Shopilots Assistant" or "I'm Shopilots Assistant".

Your task is to provide accurate, helpful, and complete answers based ONLY on the context provided below.

CRITICAL INSTRUCTIONS:
1. Read the entire context carefully before answering
2. For product questions, mention ALL relevant categories:
   AI Sales Agents:
   - Website Agent
   - Social Media Agent  
   - Messenger Agent
   - Call Agent
   - GPT Store
   
   Industry Solutions:
   - Electronics & Tech (+53% conversion rate, +41% AOV)
   - Fashion & Apparel (+49% conversion rate, +32% AOV)
   - Home & Garden (+47% conversion rate, +38% AOV)
   - Agencies & Partners
3. Include key features and benefits when discussing products
4. Be specific and include details like conversion rates, AOV improvements when mentioned in context
5. Format lists clearly using bullet points
6. If information is not in the context, politely state you don't have that specific information
7. Be conversational but professional
8. Provide complete answers, not fragments"""
    
    user_prompt = f"""Context about Shopilots:
{context_text}

Customer Question: {question}

Provide a clear, complete answer based on the context above:"""
    
    try:
        if use_openai and openai_client:
            try:
                stream = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True,
                    temperature=0.7,
                    max_tokens=500
                )
                
                # Try to get first chunk to validate the stream
                first_chunk = next(stream, None)
                if first_chunk is None:
                    raise Exception("Empty stream from OpenAI")
                
                # Process first chunk
                if first_chunk.choices and len(first_chunk.choices) > 0:
                    delta = first_chunk.choices[0].delta
                    if delta and hasattr(delta, 'content') and delta.content:
                        token = delta.content
                        yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Process remaining chunks
                for chunk in stream:
                    try:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta and hasattr(delta, 'content') and delta.content:
                                token = delta.content
                                yield f"data: {json.dumps({'token': token})}\n\n"
                    except Exception:
                        continue
                
                yield "data: [DONE]\n\n"
                return
            except Exception as openai_error:
                # Fallback to format_response if OpenAI fails
                error_msg = str(openai_error)
                print(f"OpenAI API error: {error_msg}, falling back to format_response")
                # Don't yield error, just fallback silently
                answer = format_response(docs, question) if docs else "I couldn't find specific information about that."
                for i in range(0, len(answer), 5):
                    yield f"data: {json.dumps({'token': answer[i:i+5]})}\n\n"
                yield "data: [DONE]\n\n"
                return
        
        # Fallback to local LLM
        if llm:
            streamer = llm.start_answer_iterator_streamer(f"{system_prompt}\n\n{user_prompt}", max_new_tokens=400)
            
            for token in streamer:
                try:
                    parsed_token = llm.parse_token(token)
                    if parsed_token:
                        token_str = str(parsed_token)
                        if token_str.strip():
                            yield f"data: {json.dumps({'token': token_str})}\n\n"
                except Exception:
                    continue
            
            yield "data: [DONE]\n\n"
            return
        
        answer = format_response(docs, question)
        for i in range(0, len(answer), 5):
            yield f"data: {json.dumps({'token': answer[i:i+5]})}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/api/chat/stream")
async def chat_stream_post(request: ChatRequest):
    """Streaming chat endpoint (POST)"""
    start_time = time.time()
    session_id = request.session_id or "default"
    
    # Track question event
    analytics_tracker.track_event(ConversationEvent(
        timestamp=datetime.now().isoformat(),
        session_id=session_id,
        event_type='question',
        question=request.question,
    ))
    
    if not index:
        analytics_tracker.track_event(ConversationEvent(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            event_type='error',
            question=request.question,
            error='Vector database not initialized'
        ))
        async def error_stream():
            yield f"data: {json.dumps({'error': 'Vector database not initialized'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    try:
        docs, sources = retrieve_docs(request.question, k=4)
        if not docs and "product" in request.question.lower():
            docs, sources = retrieve_docs("shopilots AI sales agents products", k=3)
        
        # Track fallback if no docs found
        if not docs:
            analytics_tracker.track_event(ConversationEvent(
                timestamp=datetime.now().isoformat(),
                session_id=session_id,
                event_type='fallback',
                question=request.question,
                error='No relevant documents found',
                docs_retrieved=0
            ))
        
        # Wrap the generator to track response time and answer
        async def tracked_stream():
            answer_chunks = []
            model_used = "openai" if use_openai else "llama-3.2-3b"
            
            try:
                # generate_streaming_response is an async generator
                async for chunk in generate_streaming_response(request.question, docs):
                    try:
                        if chunk.startswith('data: '):
                            data = chunk[6:].strip()
                            if data and data != '[DONE]':
                                try:
                                    parsed = json.loads(data)
                                    if parsed.get('token'):
                                        answer_chunks.append(parsed['token'])
                                except:
                                    pass
                        yield chunk
                    except Exception as chunk_error:
                        print(f"Error yielding chunk: {chunk_error}")
                        yield f"data: {json.dumps({'error': str(chunk_error)})}\n\n"
                        break
                
                # Track answer event
                response_time_ms = (time.time() - start_time) * 1000
                full_answer = ''.join(answer_chunks)
                detected_category = detect_product_category(request.question, full_answer)
                
                analytics_tracker.track_event(ConversationEvent(
                    timestamp=datetime.now().isoformat(),
                    session_id=session_id,
                    event_type='answer',
                    question=request.question,
                    answer=full_answer[:500],  # Store first 500 chars
                    response_time_ms=response_time_ms,
                    docs_retrieved=len(docs),
                    sources=sources[:3] if sources else None,
                    model_used=model_used,
                    answer_length=len(full_answer),
                    product_category=detected_category
                ))
            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                analytics_tracker.track_event(ConversationEvent(
                    timestamp=datetime.now().isoformat(),
                    session_id=session_id,
                    event_type='error',
                    question=request.question,
                    error=str(e),
                    response_time_ms=response_time_ms
                ))
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            tracked_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        error_message = str(e)
        analytics_tracker.track_event(ConversationEvent(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            event_type='error',
            question=request.question,
            error=error_message,
            response_time_ms=response_time_ms
        ))
        async def error_stream():
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    session_id = request.session_id or "default"
    
    # Track question event
    analytics_tracker.track_event(ConversationEvent(
        timestamp=datetime.now().isoformat(),
        session_id=session_id,
        event_type='question',
        question=request.question,
    ))
    
    if not index:
        analytics_tracker.track_event(ConversationEvent(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            event_type='error',
            question=request.question,
            error='Vector database not initialized'
        ))
        raise HTTPException(status_code=500, detail="Vector database not initialized")
    
    try:
        docs, sources = retrieve_docs(request.question, k=4)
        
        # Ensure we have docs - if not, try product-specific search
        if not docs and "product" in request.question.lower():
            docs, sources = retrieve_docs("shopilots AI sales agents products", k=3)
        
        # Track fallback if no docs found
        if not docs:
            analytics_tracker.track_event(ConversationEvent(
                timestamp=datetime.now().isoformat(),
                session_id=session_id,
                event_type='fallback',
                question=request.question,
                error='No relevant documents found',
                docs_retrieved=0
            ))
        
        model_used = None
        answer = None
        
        # Always use LLM for better responses when available
        if components_loaded and llm and docs:
            try:
                chat_history = get_chat_history(request.session_id)
                
                # Create enhanced prompt with system instructions
                context_text = "\n\n".join([doc.page_content for doc in docs[:3]])
                
                enhanced_prompt = f"""You are Shopilots Assistant, a helpful AI assistant for Shopilots, an AI-powered e-commerce sales platform. When introducing yourself, say "I am Shopilots Assistant" or "I'm Shopilots Assistant".

Your task is to provide accurate, helpful, and complete answers based ONLY on the context provided below.

CRITICAL INSTRUCTIONS:
1. Read the entire context carefully before answering
2. For product questions, mention ALL relevant categories:
   AI Sales Agents:
   - Website Agent
   - Social Media Agent  
   - Messenger Agent
   - Call Agent
   - GPT Store
   
   Industry Solutions:
   - Electronics & Tech (+53% conversion rate, +41% AOV)
   - Fashion & Apparel (+49% conversion rate, +32% AOV)
   - Home & Garden (+47% conversion rate, +38% AOV)
   - Agencies & Partners
3. Include key features and benefits when discussing products
4. Be specific and include details like conversion rates, AOV improvements when mentioned in context
5. Format lists clearly using bullet points
6. If information is not in the context, politely state you don't have that specific information
7. Be conversational but professional
8. Provide complete answers, not fragments

Context about Shopilots:
{context_text}

Customer Question: {request.question}

Provide a clear, complete answer based on the context above:"""
                
                model_used = "llama-3.2-3b"
                answer = llm.generate_answer(enhanced_prompt, max_new_tokens=350)
                
                answer = answer.strip()
                prefixes_to_remove = ["Refined Answer:", "Answer:", "Response:", "Based on", "According to"]
                for prefix in prefixes_to_remove:
                    if answer.startswith(prefix):
                        answer = answer[len(prefix):].strip()
                
                # Ensure minimum length
                if len(answer) < 30:
                    # Retry with streaming for better generation
                    streamer = llm.start_answer_iterator_streamer(enhanced_prompt, max_new_tokens=350)
                    answer = ""
                    token_count = 0
                    for token in streamer:
                        parsed = llm.parse_token(token)
                        answer += parsed
                        token_count += 1
                        if token_count > 250 or len(answer) > 700:
                            break
                    answer = answer.strip()
                
                if answer and len(answer) > 20:
                    chat_history.append(f"question: {request.question}, answer: {answer}")
                    response_time_ms = (time.time() - start_time) * 1000
                    detected_category = detect_product_category(request.question, answer)
                    
                    # Track answer event
                    analytics_tracker.track_event(ConversationEvent(
                        timestamp=datetime.now().isoformat(),
                        session_id=session_id,
                        event_type='answer',
                        question=request.question,
                        answer=answer[:500],
                        response_time_ms=response_time_ms,
                        docs_retrieved=len(docs),
                        sources=sources[:3] if sources else None,
                        model_used=model_used,
                        answer_length=len(answer),
                        product_category=detected_category
                    ))
                    
                    return ChatResponse(
                        answer=answer,
                        sources=[{"document": s.get("document", "Unknown"), "score": s.get("score", 0)} for s in sources[:2]]
                    )
            except Exception as e:
                print(f"LLM error: {e}")
                analytics_tracker.track_event(ConversationEvent(
                    timestamp=datetime.now().isoformat(),
                    session_id=session_id,
                    event_type='error',
                    question=request.question,
                    error=f"LLM error: {str(e)}",
                    response_time_ms=(time.time() - start_time) * 1000
                ))
                pass
        
        # Fallback to formatted response
        answer = format_response(docs, request.question)
        response_time_ms = (time.time() - start_time) * 1000
        detected_category = detect_product_category(request.question, answer)
        
        # Track fallback response
        if not model_used:
            analytics_tracker.track_event(ConversationEvent(
                timestamp=datetime.now().isoformat(),
                session_id=session_id,
                event_type='fallback',
                question=request.question,
                answer=answer[:500],
                response_time_ms=response_time_ms,
                docs_retrieved=len(docs) if docs else 0,
                sources=sources[:3] if sources else None,
                error='Using formatted response fallback',
                answer_length=len(answer),
                product_category=detected_category
            ))
        else:
            # Track answer even if it's short
            analytics_tracker.track_event(ConversationEvent(
                timestamp=datetime.now().isoformat(),
                session_id=session_id,
                event_type='answer',
                question=request.question,
                answer=answer[:500],
                response_time_ms=response_time_ms,
                docs_retrieved=len(docs) if docs else 0,
                sources=sources[:3] if sources else None,
                model_used=model_used or "formatted",
                answer_length=len(answer),
                product_category=detected_category
            ))
        
        return ChatResponse(
            answer=answer,
            sources=[{"document": s.get("document", "Unknown"), "score": s.get("score", 0)} for s in sources[:2]]
        )
            
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        analytics_tracker.track_event(ConversationEvent(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            event_type='error',
            question=request.question,
            error=str(e),
            response_time_ms=response_time_ms
        ))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    initialize_components()
    uvicorn.run(app, host="0.0.0.0", port=8080)
