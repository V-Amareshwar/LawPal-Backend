# --- build_db.py ---
# This is a one-time script. Run this LOCALLY on your computer
# or in the deployment environment's console ONCE.

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import os


print("--- Starting Database Build Process ---")

# --- 1. Load Data from Phase A ---
data_filename = 'processed_legal_data.json'
with open(data_filename, 'r', encoding='utf-8') as f:
    all_chunks = json.load(f)

# --- 2. Build and Save Vector Database (ChromaDB) ---
print("Initializing embedding model (BAAI/bge-large-en-v1.5)...")
# This will run on your CPU. It will be slow, but it only has to be done once.
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

print("Initializing persistent ChromaDB...")
# This creates a persistent database on your disk in a folder named 'chroma_db'
db_path = "chroma_db"
if os.path.exists(db_path):
    import shutil
    shutil.rmtree(db_path) # Clear old database if it exists
client = chromadb.PersistentClient(path=db_path)

collection_name = "indian_legal_docs_final"
collection = client.create_collection(name=collection_name)

print(f"Embedding and indexing all {len(all_chunks)} chunks. This will take a while...")
documents = [chunk['chunk_text'] for chunk in all_chunks]
metadatas = [{'source': chunk['source_document'], 'section': chunk['section_id']} for chunk in all_chunks]
ids = [chunk['chunk_id'] for chunk in all_chunks]
batch_size = 32
for i in range(0, len(documents), batch_size):
    print(f"  - Embedding batch {i//batch_size + 1}...")
    collection.add(
        ids=ids[i:i+batch_size],
        documents=documents[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size],
        embeddings=embedding_model.encode(documents[i:i+batch_size]).tolist()
    )
print("✅ Vector Database created and saved to 'chroma_db' folder.")

# --- 3. Build and Save BM25 Index ---
print("\nInitializing and saving BM25 Index...")
tokenized_corpus = [doc.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Save the bm25 object to a file using pickle
with open('bm25_index.pkl', 'wb') as f:
    pickle.dump(bm25, f)

# Save the 'ids' list as well, as it's needed by the search function
with open('chunk_ids.json', 'w') as f:
    json.dump(ids, f)
    
print("✅ BM25 Index saved to 'bm25_index.pkl'.")
print("\n--- DATABASE BUILD COMPLETE ---")