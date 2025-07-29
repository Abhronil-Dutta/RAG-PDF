

import os
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from tqdm import tqdm

# Device setup: use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model setup
MODEL_NAME = "intfloat/e5-small-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

def embed_text(texts, batch_size=16):
    """
    Encode a list of texts into embeddings using e5-small-v2.
    Returns numpy array of shape (n_texts, emb_dim).
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch_texts = texts[i:i+batch_size]
        # e5 expects input to start with 'passage: '
        batch_inputs = ["passage: " + t.strip().replace("\n", " ") for t in batch_texts]
        inputs = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model_output = model(**inputs)
            # Mean pooling
            emb = model_output.last_hidden_state.mean(dim=1)
            emb = emb.cpu().numpy()
            embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    # Normalize to unit vectors (important for cosine similarity in FAISS)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    return embeddings

def build_faiss_index(embeddings):
    """
    Build a FAISS index (Inner Product for cosine similarity) for fast vector search.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index

def save_index(index, out_path):
    faiss.write_index(index, out_path)

def save_metadata(metadata, out_path):
    with open(out_path, "wb") as f:
        pickle.dump(metadata, f)

def store_embeddings(chunks_dict, emb_dir="E:\\RAG\\embeddings"):
    """
    Given {doc: [chunks]}, generate embeddings, build index, and save metadata.
    """
    os.makedirs(emb_dir, exist_ok=True)
    all_chunks = []
    chunk_sources = []  # To keep track of which doc/chunk each embedding is from
    for doc, chunks in chunks_dict.items():
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_sources.append((doc, idx))
    print(f"Total chunks to embed: {len(all_chunks)}")
    embeddings = embed_text(all_chunks)
    index = build_faiss_index(embeddings)
    save_index(index, os.path.join(emb_dir, "faiss.index"))
    save_metadata({"chunks": all_chunks, "sources": chunk_sources}, os.path.join(emb_dir, "metadata.pkl"))
    print(f"Saved FAISS index and metadata in {emb_dir}")

if __name__ == "__main__":
    # Example usage: load chunked docs from previous step
    import json

    # Suppose you saved the chunked output as chunked.json:
    # {doc_name: [chunk1, chunk2, ...]}
    chunked_path = "chunked.json"
    with open(chunked_path, "r", encoding="utf-8") as f:
        chunked_docs = json.load(f)
    store_embeddings(chunked_docs)
