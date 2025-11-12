# --- app.py ---
# This is your final, fast-starting backend server.

import numpy as np
from dotenv import load_dotenv
import os
import json
import re
import pickle
from flask import Flask, request, jsonify
from groq import Groq, RateLimitError  # <-- MODIFIED: Import RateLimitError
import tiktoken
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import chromadb
from rank_bm25 import BM25Okapi

load_dotenv()

print("--- Initializing AI Legal Assistant Backend ---")

# --- GLOBAL VARIABLES ---
# We load all models and databases ONCE at startup, not per-request.

# --- 1. Load Groq API Keys & Rotation State ---
print("Loading Groq API keys...")
GROQ_API_KEYS = os.environ.get("GROQ_API_KEYS", "").split(',')
GROQ_API_KEYS = [key.strip() for key in GROQ_API_KEYS if key.strip()]  # Clean up keys

if not GROQ_API_KEYS:
    print("❌ FATAL ERROR: GROQ_API_KEYS not found in .env file.")
    exit()

print(f"✅ Loaded {len(GROQ_API_KEYS)} Groq API keys.")

# --- ADDED: State file to remember the last used key index ---
API_KEY_STATE_FILE = "api_key_tracker.json"

def load_key_index():
    """Loads the next key index from the state file."""
    try:
        with open(API_KEY_STATE_FILE, 'r') as f:
            data = json.load(f)
            # Modulo to ensure index is always valid, even if keys were removed
            return data.get("next_key_index", 0) % len(GROQ_API_KEYS)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0  # Start from the first key if file is missing or corrupt

def save_key_index(index):
    """Saves the next key index to the state file."""
    with open(API_KEY_STATE_FILE, 'w') as f:
        json.dump({"next_key_index": index}, f)

# --- (End of new Groq logic) ---


# 2. Load Embedding Model
print("Loading embedding model (BAAI/bge-large-en-v1.5)...")
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
print("✅ Embedding model loaded.")

# 3. Load Persistent Vector Database
print("Loading persistent Vector Database (ChromaDB)...")
client_db = chromadb.PersistentClient(path="chroma_db")
collection = client_db.get_collection(name="indian_legal_docs_final")
print("✅ Vector Database loaded.")

# 4. Load BM25 Index
print("Loading BM25 Index...")
with open('bm25_index.pkl', 'rb') as f:
    bm25 = pickle.load(f)
with open('chunk_ids.json', 'r') as f:
    ids = json.load(f)
print("✅ BM25 Index loaded.")

# 5. Load all chunk data for final retrieval
print("Loading all chunk data...")
with open('processed_legal_data.json', 'r', encoding='utf-8') as f:
    all_chunks = {chunk['chunk_id']: chunk for chunk in json.load(f)}
print("✅ All data loaded.")

# 6. Load Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# --- HELPER FUNCTIONS (Copied from our notebook) ---
# ... (Your hybrid_search, advanced_legal_reranker, count_tokens, 
#      and build_polished_prompt functions are all unchanged) ...

def hybrid_search(query, top_k=20):
    print(f"\n   - Executing hybrid search for query: '{query}'")
    vector_results = collection.query(
        query_embeddings=embedding_model.encode([query]).tolist(), 
        n_results=top_k
    )
    vector_ids = vector_results['ids'][0]
    print(f"     - Vector search found {len(vector_ids)} results.")
    
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_ids = [ids[i] for i in top_n_indices]
    print(f"     - BM25 search found {len(bm25_ids)} results.")
    
    fused_ids = list(dict.fromkeys(vector_ids + bm25_ids))
    print(f"     - Fused results to {len(fused_ids)} unique chunks.")
    
    fused_results_data = [all_chunks[chunk_id] for chunk_id in fused_ids if chunk_id in all_chunks]
    return fused_results_data

def advanced_legal_reranker(query, search_results, embedding_model):
    print("\n   - Applying Final Legal Reranker...")
    priority_docs = ["Constitution of India.json", "Indian Penal Code, 1860.json"]
    query_embedding = embedding_model.encode(query)
    chunk_texts = [res['chunk_text'] for res in search_results]
    chunk_embeddings = embedding_model.encode(chunk_texts)
    similarities = cos_sim(query_embedding, chunk_embeddings)[0]

    query_doc_match = re.search(r'(indian evidence act|indian penal code|code of criminal procedure|code of civil procedure|constitution of india|motor vehicles act)', query, re.IGNORECASE)
    query_sec_match = re.search(r'(section|article) (\d+\w*)', query, re.IGNORECASE)
    query_doc = query_doc_match.group(0).lower() if query_doc_match else None
    query_sec_num_str = query_sec_match.group(2) if query_sec_match else None
    
    keywords = ["insurance", "penalty", "offense", "fine", "imprisonment", "uninsured", "expired", "alcohol", "blood"]
    query_keywords = [word for word in query.lower().split() if word in keywords]

    final_results = []
    for i, chunk in enumerate(search_results):
        heuristic_score, semantic_score, exact_match_boost, keyword_boost = 1.0, 0.0, 0.0, 0.0
        if chunk['source_document'] in priority_docs: heuristic_score = 1.5
        semantic_score = similarities[i].item()
        
        try:
            chunk_doc = chunk['source_document'].lower()
            match = re.search(r'\d+', chunk['section_id'])
            if match:
                chunk_sec_num = int(match.group())
                if query_doc and query_sec_num_str and query_doc in chunk_doc and str(chunk_sec_num) == query_sec_num_str:
                    exact_match_boost = 10.0
        except: pass
        
        chunk_text_lower = chunk['chunk_text'].lower()
        for keyword in query_keywords:
            if keyword in chunk_text_lower: keyword_boost += 0.1
        
        weight_heuristic, weight_semantic, weight_keyword = 0.1, 0.7, 0.2
        base_score = (heuristic_score * weight_heuristic) + (semantic_score * weight_semantic) + (keyword_boost * weight_keyword)
        final_score = base_score + exact_match_boost
        
        final_results.append({"final_score": round(final_score, 4), "chunk_data": chunk})
        
    final_results.sort(key=lambda x: x['final_score'], reverse=True)
    print("   - Reranking complete.")
    return final_results

def count_tokens(text):
    return len(tokenizer.encode(text))

def build_polished_prompt(query, reranked_chunks, token_budget):
    system_prompt = """
    **SYSTEM PROMPT – AI LAWYER CHAT (Frontend Version)**
    You are a professional Indian Legal AI Assistant. Answer the user's legal questions strictly based on the provided sources (Acts, Rules, or Case Law excerpts).
    ### Guidelines:
    1. **Source-Based Responses**
       - Use only the information from the provided sources.
       - Do not invent or assume facts.
       - If insufficient information is available, respond:
         "The available legal documents do not contain sufficient information to answer this question."
    2. **Legal Reasoning**
       - Identify relevant Sections or Clauses and explain why they apply.
       - Integrate multiple sources smoothly if necessary.
    3. **Citation Style**
       - Cite naturally: “Under Section 13B of the Hindu Marriage Act, 1955…”
       - Avoid numeric tags like [Source 1] or URLs.
    4. **Tone & Style**
       - Professional, clear, and readable.
       - Avoid long academic lists or repetitive wording.
       - Keep answers concise for chat display, but legally precise.
    5. **Confidence**
       - Optionally, include a confidence rating at the end:
         *Confidence: X/10 — based on the completeness of the provided legal sources.*
    6. **Applicability**
       - This prompt works for any Indian legal dataset.
    Remember: You are a legal reasoning assistant, not a general explainer.
    """
    prompt_header_tokens = count_tokens(system_prompt + query)
    current_tokens = prompt_header_tokens
    context_str = ""
    source_count = 0
    print("   - Packing context within token budget...")
    for item in reranked_chunks:
        chunk_data = item['chunk_data']
        next_chunk_str = f"Source {source_count + 1}:\n"
        next_chunk_str += f"  Document: {chunk_data['source_document']}\n"
        next_chunk_str += f"  Section: {chunk_data['section_id']}\n"
        next_chunk_str += f"  Text: {chunk_data['chunk_text']}\n\n"
        chunk_tokens = count_tokens(next_chunk_str)
        if current_tokens + chunk_tokens <= token_budget:
            context_str += next_chunk_str
            current_tokens += chunk_tokens
            source_count += 1
        else:
            print(f"   - Token budget reached. Packed {source_count} sources.")
            break
    final_prompt = f"{system_prompt}\n---\n**Sources:**\n{context_str}---\n**User's Question:**\n{query}"
    return final_prompt, current_tokens

# --- FLASK SERVER LOGIC ---
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "AI Legal Backend is running!"})

@app.route("/ask", methods=["POST"])
def process_query():
    """API endpoint to run the full RAG pipeline."""
    try:
        query = request.get_json().get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        print(f"\n--- Received Query --- \nQuery: '{query}'")
        
        # 1. Run Retrieval & Reranking (Unchanged)
        print("1. Retrieving and reranking context...")
        fused_results = hybrid_search(query=query, top_k=20)
        final_reranked_chunks = advanced_legal_reranker(
            query=query,
            search_results=fused_results,
            embedding_model=embedding_model
        )

        # 2. Build Prompt (Unchanged)
        print("2. Building prompt...")
        TOKEN_BUDGET = 8000
        final_prompt, _ = build_polished_prompt(
            query,
            final_reranked_chunks,
            TOKEN_BUDGET
        )

        # --- 3. Call Groq API (MODIFIED with Rotation Logic) ---
        print(f"3. Calling llama-3.3-70b-versatile with key rotation...")
        
        start_index = load_key_index()
        num_keys = len(GROQ_API_KEYS)
        final_answer = None
        
        # Try each key once, starting from the last known index
        for i in range(num_keys):
            current_index = (start_index + i) % num_keys
            current_key = GROQ_API_KEYS[current_index]
            
            print(f"   - Attempting with API Key Index: {current_index}")
            
            try:
                # Initialize client *inside the loop* with the specific key
                client = Groq(api_key=current_key)
                
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": final_prompt}],
                    model="llama-3.3-70b-versatile",
                )
                final_answer = chat_completion.choices[0].message.content
                
                # --- SUCCESS ---
                print("✅ Answer generated successfully.")
                
                # Save the *next* index for the next request
                save_key_index((current_index + 1) % num_keys)
                
                break  # Exit the loop on success

            except RateLimitError as e:
                print(f"   - ⚠️ Rate limit exceeded for Key Index {current_index}. Trying next key...")
                # The loop will continue to the next key
            
            except Exception as e:
                # This is a different, unexpected error
                print(f"❌ An unexpected error occurred with Key Index {current_index}: {e}")
                # We re-raise this error to be caught by the outer try/except block
                raise e

        # If all keys were tried and all failed (e.g., all are rate-limited)
        if final_answer is None:
            print("❌ All API keys failed (all are rate-limited).")
            return jsonify({"error": "All API keys are currently rate limited. Please try again later."}), 503

        # --- (End of modified Groq logic) ---
        
        return jsonify({"answer": final_answer})

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# This is the standard way to run a Flask app with a production server like gunicorn
if __name__ == "__main__":
    # Get the port from the environment, defaulting to 8080 (common for free platforms)
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)