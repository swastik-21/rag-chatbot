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

sys.path.insert(0, str(Path(__file__).parent / "chatbot"))

from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
from bot.conversation.conversation_handler import answer_with_context, refine_question
from bot.conversation.ctx_strategy import get_ctx_synthesis_strategy
from bot.model.model_registry import Model, get_model_settings
import os
from openai import OpenAI

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
    
    try:
        embedding = Embedder()
        index = Chroma(persist_directory=str(vector_store_path), embedding=embedding)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                openai_client = OpenAI(api_key=api_key)
                use_openai = True
                components_loaded = True
                print("Using OpenAI API")
                return True
            except Exception as e:
                print(f"OpenAI init failed: {e}, falling back to local model")
        
        try:
            model_name = Model.LLAMA_3_2_three.value
            model_settings = get_model_settings(model_name)
            model_folder.mkdir(parents=True, exist_ok=True)
            llm = LamaCppClient(model_folder=model_folder, model_settings=model_settings)
            synthesis_strategy = get_ctx_synthesis_strategy("create-and-refine", llm=llm)
            components_loaded = True
            use_openai = False
            print("Using local LLM (Llama 3.2 3B)")
        except Exception:
            components_loaded = False
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def get_chat_history(session_id: str) -> ChatHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatHistory(total_length=6)
    return chat_histories[session_id]

def retrieve_docs(query: str, k: int = 5):
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
    
    # Extract products
    if any(word in query_lower for word in ["product", "offer", "provide", "what do you", "services"]):
        products = []
        
        if "website agent" in all_text:
            products.append("• Website Agent - Embed on your e-commerce site. Deploy conversational salesforce that sell — not just chat.")
        if "social media agent" in all_text:
            products.append("• Social Media Agent - Engage on social platforms with AI-powered shopping assistance.")
        if "messenger agent" in all_text:
            products.append("• Messenger Agent - Chat on WhatsApp & Messenger. Handle customer inquiries and drive sales through messaging platforms.")
        if "call agent" in all_text:
            products.append("• Call Agent - Handle customer phone calls with AI voice capabilities.")
        if "gpt store" in all_text:
            products.append("• GPT Store - Launch in ChatGPT & GPT Store. Create a custom GPT shopping assistant.")
        
        if products:
            return "Shopilots offers the following AI Sales Agents:\n\n" + "\n".join(products)
    
    # Extract platforms
    elif any(word in query_lower for word in ["platform", "integrate", "shopify", "woocommerce", "supported"]):
        platforms = []
        
        if "shopify" in all_text:
            platforms.append("• Shopify")
        if "woocommerce" in all_text:
            platforms.append("• WooCommerce")
        if "magento" in all_text:
            platforms.append("• Magento")
        if "whatsapp" in all_text or "messenger" in all_text:
            platforms.append("• WhatsApp & Messenger")
        
        if platforms:
            return "Shopilots integrates with:\n\n" + "\n".join(platforms)
    
    # Extract pricing
    elif any(word in query_lower for word in ["price", "pricing", "plan", "cost"]):
        return "Shopilots offers:\n\n• Performance Plan - Free to start (up to 500 conversations/month)\n• Business Plan - $299/month (up to 5,000 conversations/month)\n• Enterprise Plan - Custom pricing (unlimited conversations, dedicated support)"
    
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
                <div class="message-content">Hello! I can help you with questions about Shopilots products, platforms, and services.</div>
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
    
    system_prompt = """You are a knowledgeable customer service representative for Shopilots, an AI-powered e-commerce sales platform.

Your task is to provide accurate, helpful, and complete answers based ONLY on the context provided below.

CRITICAL INSTRUCTIONS:
1. Read the entire context carefully before answering
2. For product questions, mention ALL relevant AI Sales Agents:
   - Website Agent
   - Social Media Agent  
   - Messenger Agent
   - Call Agent
   - GPT Store
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
    if not index:
        async def error_stream():
            yield f"data: {json.dumps({'error': 'Vector database not initialized'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    try:
        docs, _ = retrieve_docs(request.question, k=4)
        if not docs and "product" in request.question.lower():
            docs, _ = retrieve_docs("shopilots AI sales agents products", k=3)
        
        return StreamingResponse(
            generate_streaming_response(request.question, docs),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        async def error_stream():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not index:
        raise HTTPException(status_code=500, detail="Vector database not initialized")
    
    try:
        docs, sources = retrieve_docs(request.question, k=4)
        
        # Ensure we have docs - if not, try product-specific search
        if not docs and "product" in request.question.lower():
            docs, sources = retrieve_docs("shopilots AI sales agents products", k=3)
        
        # Always use LLM for better responses when available
        if components_loaded and llm and docs:
            try:
                chat_history = get_chat_history(request.session_id)
                
                # Create enhanced prompt with system instructions
                context_text = "\n\n".join([doc.page_content for doc in docs[:3]])
                
                enhanced_prompt = f"""You are a knowledgeable customer service representative for Shopilots, an AI-powered e-commerce sales platform.

Your task is to provide accurate, helpful, and complete answers based ONLY on the context provided below.

CRITICAL INSTRUCTIONS:
1. Read the entire context carefully before answering
2. For product questions, mention ALL relevant AI Sales Agents:
   - Website Agent
   - Social Media Agent  
   - Messenger Agent
   - Call Agent
   - GPT Store
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
                    return ChatResponse(
                        answer=answer,
                        sources=[{"document": s.get("document", "Unknown"), "score": s.get("score", 0)} for s in sources[:2]]
                    )
            except Exception as e:
                print(f"LLM error: {e}")
                pass
        
        # Fallback to formatted response
        answer = format_response(docs, request.question)
        return ChatResponse(
            answer=answer,
            sources=[{"document": s.get("document", "Unknown"), "score": s.get("score", 0)} for s in sources[:2]]
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    initialize_components()
    uvicorn.run(app, host="0.0.0.0", port=8080)
