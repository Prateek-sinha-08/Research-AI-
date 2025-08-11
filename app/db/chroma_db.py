# app/db/chroma_db.py

import chromadb

# ✅ Persistent ChromaDB client (no embedding_function here)
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

def get_or_create_collection(name: str):
    """
    Get or create a ChromaDB collection without an automatic embedding function.
    We'll pass precomputed embeddings manually.
    """
    return chroma_client.get_or_create_collection(name=name)
