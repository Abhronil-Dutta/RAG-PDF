# query_engine.py

import pickle
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# === CONFIG ===
EMBEDDING_MODEL = "intfloat/e5-small-v2"
EMBEDDINGS_DIR = "E:\\RAG\\embeddings"
OLLAMA_URL = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen:latest"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)

def embed_query(query):
    # e5 expects "query: " prefix for queries!
    text = "query: " + query.strip().replace("\n", " ")
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        model_output = model(**inputs)
        emb = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
    # Normalize
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
    return emb.astype(np.float32)

def load_faiss_and_metadata(emb_dir=EMBEDDINGS_DIR):
    # Load FAISS index
    index = faiss.read_index(f"{emb_dir}/faiss.index")
    # Load metadata
    with open(f"{emb_dir}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search_index(query_emb, index, top_k=5):
    # query_emb shape: (1, dim)
    D, I = index.search(query_emb, top_k)
    return I[0], D[0]  # Indices and scores

def build_prompt(chunks, query, max_context_len=2000):
    # Concatenate top chunks, keeping length under max_context_len
    context = ""
    for chunk in chunks:
        if len(context) + len(chunk) > max_context_len:
            break
        context += chunk + "\n"
    prompt = f"""Use the context below to answer the question as accurately as possible.

Context:
{context}

Question: {query}
Answer:"""
    return prompt

def ask_ollama_qwen(prompt, model=QWEN_MODEL, ollama_url=OLLAMA_URL):
    # Call Ollama API with the prompt
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(ollama_url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    # 'response' or 'message' key can vary by Ollama version; handle both
    return data.get("response") or data.get("message")

def rag_query(query, top_k=5):
    # Load index and metadata
    index, metadata = load_faiss_and_metadata()
    # Embed the user query
    query_emb = embed_query(query)
    # Search for similar chunks
    indices, scores = search_index(query_emb, index, top_k=top_k)
    # Retrieve corresponding context chunks
    top_chunks = [metadata["chunks"][i] for i in indices]
    # Build prompt for Qwen
    prompt = build_prompt(top_chunks, query)
    # Get response from Qwen via Ollama
    answer = ask_ollama_qwen(prompt)
    return answer, top_chunks

if __name__ == "__main__":
    print("=== Retrieval-Augmented Generation (RAG) with Qwen & Ollama ===")
    user_query = input("Enter your question: ")
    answer, context = rag_query(user_query)
    print("\n--- Qwen Answer ---")
    print(answer)
    print("\n--- Retrieved Context Chunks ---")
    for idx, chunk in enumerate(context, 1):
        print(f"\n[Chunk {idx}]:\n{chunk}\n")
