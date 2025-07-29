# Retrieval-Augmented Generation (RAG) with Qwen & Ollama

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows you to query the contents of a collection of PDF documents using natural language. It combines state-of-the-art embedding models, vector search, and large language models (LLMs) to provide accurate, context-aware answers, with the Qwen LLM served via [Ollama](https://ollama.com/).

## Features

- **PDF Extraction:** Automatically extracts text from all PDFs in a specified directory.
- **Text Chunking:** Splits extracted text into overlapping, context-preserving chunks using LangChain.
- **Embeddings:** Generates vector embeddings for each chunk using the `intfloat/e5-small-v2` model.
- **Vector Search:** Stores embeddings in a FAISS index for fast similarity search.
- **RAG Query Engine:** Retrieves the most relevant chunks for a user query and constructs a prompt for the Qwen LLM.
- **LLM Integration:** Sends the prompt to Qwen via Ollama and returns a contextually grounded answer.
- **Interactive CLI:** Step-by-step pipeline with an interactive query mode.

## Workflow

1. **Extract Text from PDFs:**  
   All PDFs in the `data/` directory are processed and their text is extracted.

2. **Chunk Text:**  
   Extracted texts are split into overlapping chunks for better context retrieval.

3. **Generate Embeddings:**  
   Each chunk is embedded into a vector space using a transformer model.

4. **Build Vector Store:**  
   Embeddings are indexed using FAISS for efficient similarity search.

5. **Query:**
   - User enters a natural language question.
   - The system retrieves the most relevant chunks.
   - A prompt is built and sent to Qwen via Ollama.
   - The answer and supporting context are displayed.

## Setup

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) running locally with the Qwen model pulled (`ollama pull qwen`)
- CUDA-enabled GPU (optional, for faster embedding generation)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd RAG
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**

   - Place your PDF files in the `data/` directory.

4. **Start Ollama with Qwen:**
   - Make sure Ollama is running and the Qwen model is available.

### Running the Pipeline

```bash
python main.py
```

This will:

- Extract text from PDFs
- Chunk the text
- Generate and store embeddings
- Launch an interactive query session

## Example Usage

```
=== Interactive RAG Query ===

Enter your question (or type 'exit'): What is the mission of Shoolini University?
--- Answer ---
<LLM-generated answer>
--- Retrieved Context Chunks ---
[Chunk 1]:
<Relevant text from your PDFs>
```

## Project Structure

```
RAG/
  ├── data/                # Place your PDFs here
  ├── embeddings/          # Stores FAISS index and metadata
  ├── src/
  │   ├── load_pdf.py      # PDF extraction
  │   ├── chunk_text.py    # Text chunking
  │   ├── embed_store.py   # Embedding and vector store
  │   ├── query_engine.py  # RAG query logic
  │   └── utils.py         # Utilities
  ├── main.py              # Pipeline entry point
  ├── requirements.txt
```

## Dependencies

- `pymupdf` (PDF extraction)
- `langchain` (text chunking)
- `torch`, `transformers` (embeddings)
- `faiss-cpu` (vector search)
- `tqdm` (progress bars)
- `requests` (Ollama API calls)

## Notes

- The embedding model and LLM can be swapped with minimal code changes.
- For best results, use a GPU for embedding generation.
- Ollama must be running locally and accessible at `http://localhost:11434`.
