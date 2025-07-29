# load_pdf.py

import os
import fitz  # PyMuPDF

def list_pdfs(data_dir):
    """
    List all PDF files in the data directory.
    """
    return [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a single PDF file.
    """
    text = []
    doc = fitz.open(pdf_path)  # Open the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)      # Load each page
        page_text = page.get_text()         # Extract text from the page
        text.append(page_text)              # Append to our list
    doc.close()
    return "\n".join(text)                  # Join all pages' text

def extract_all_pdfs(data_dir):
    """
    Extracts text from all PDFs in the data directory.
    Returns a dictionary: {filename: extracted_text}
    """
    pdf_files = list_pdfs(data_dir)
    pdf_texts = {}
    for pdf_file in pdf_files:
        full_path = os.path.join(data_dir, pdf_file)
        print(f"Extracting: {pdf_file}")
        text = extract_text_from_pdf(full_path)
        pdf_texts[pdf_file] = text
    return pdf_texts

if __name__ == "__main__":
   
    data_dir = "data" 
    pdf_texts = extract_all_pdfs(data_dir)
    for pdf_file, text in pdf_texts.items():
        out_path = f"{pdf_file}_extracted.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {out_path}")
