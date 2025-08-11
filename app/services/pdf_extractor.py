# app/services/pdf_extractor.py
import fitz  # PyMuPDF
from typing import Dict

def extract_pdf_data(file_bytes: bytes, filename: str) -> Dict:
    """
    Extract title, text, and metadata from a PDF file.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")

    metadata = doc.metadata

    return {
        "filename": filename,
        "title": metadata.get("title") or filename,
        "author": metadata.get("author"),
        "creationDate": metadata.get("creationDate"),
        "subject": metadata.get("subject"),
        "keywords": metadata.get("keywords"),
        "text": text,
        "text_excerpt": text[:1000]  # Show first 1000 characters for preview
    }
