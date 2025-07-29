# main.py

import os
import json
import pickle

from src.load_pdf import extract_all_pdfs
from src.chunk_text import chunk_documents
from src.embed_store import store_embeddings
from src.query_engine import rag_query

DATA_DIR = "data"
CHUNKS_JSON = "chunked.json"
EMB_DIR = "embeddings"

def extract_texts():
    print("Step 1: Extracting text from PDFs...")
    pdf_texts = extract_all_pdfs(DATA_DIR)
    print(f"Extracted text from {len(pdf_texts)} PDFs.")
    return pdf_texts

def chunk_texts(pdf_texts):
    print("Step 2: Chunking extracted text...")
    chunked = chunk_documents(pdf_texts)
    print(f"Chunked into {sum(len(v) for v in chunked.values())} total chunks.")
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)
    print(f"Chunks saved to {CHUNKS_JSON}.")
    return chunked

def embed_chunks(chunked):
    print("Step 3: Generating and storing embeddings...")
    store_embeddings(chunked, EMB_DIR)
    print("Embeddings stored.")

def interactive_query():
    print("\n=== Interactive RAG Query ===")
    while True:
        query = input("\nEnter your question (or type 'exit'): ").strip()
        if not query or query.lower() == "exit":
            print("Exiting.")
            break
        answer, context = rag_query(query)
        print("\n--- Answer ---")
        print(answer)

if __name__ == "__main__":
    pdf_texts = extract_texts()
    with open("pdf_texts.json", "w", encoding="utf-8") as f:
        json.dump(pdf_texts, f, ensure_ascii=False, indent=2)
    chunked = chunk_texts(pdf_texts)
    embed_chunks(chunked)
    interactive_query()