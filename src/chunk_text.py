# chunk_text.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Splits the input text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    Args:
        text (str): The raw text to split.
        chunk_size (int): Max size of each chunk (in characters).
        chunk_overlap (int): Number of overlapping characters between chunks.
    Returns:
        List[str]: List of chunked text segments.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

# Optional: handle multiple documents
def chunk_documents(docs_dict, chunk_size=500, chunk_overlap=50):
    """
    Splits each document's text into chunks.
    Args:
        docs_dict (dict): {doc_name: text}
    Returns:
        dict: {doc_name: [chunks]}
    """
    chunked = {}
    for name, text in docs_dict.items():
        chunked[name] = chunk_text(text, chunk_size, chunk_overlap)
    return chunked

import json

if __name__ == "__main__":
    sample_file = "E:\\RAG\\About Shoolini University.pdf_extracted.txt"
    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()
    # Use the filename (without path) as the key
    file_key = sample_file.split("\\")[-1]
    chunks = chunk_text(text)
    print(f"Total Chunks: {len(chunks)}")
    # Build dictionary
    chunks_dict = {file_key: chunks}
    # Save as JSON
    with open("chunked.json", "w", encoding="utf-8") as out:
        json.dump(chunks_dict, out, ensure_ascii=False, indent=2)
    print("Chunks saved to chunked.json")

