# app/services/embeddings.py

import logging
from uuid import uuid4
from typing import List
import tiktoken
import httpx
import asyncio
from app.db.chroma_db import get_or_create_collection
from app.utils.text_splitter import chunk_text
from app.utils.config import settings

# Setup logger
logger = logging.getLogger("uvicorn.error")

# Semantic search instruction
INSTRUCTION = "Represent the scientific research paper chunk for semantic retrieval"

# Tokenizer (Mistral uses cl100k_base like OpenAI)
encoding = tiktoken.get_encoding("cl100k_base")

# Mistral embedding API
MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"
HEADERS = {
    "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}
EMBED_MODEL = "mistral-embed"  # 1024-dim embeddings

# Safe request size
MAX_BATCH_SIZE = 12
MAX_RETRIES = 3  # Retry failed batches


def get_token_count(instr: str, chunk: str) -> int:
    """Count tokens for instruction + text chunk."""
    return len(encoding.encode(instr)) + len(encoding.encode(chunk))


def embed_chunks(text: str) -> List[str]:
    """
    Split text into chunks within 512-token limit (including instruction).
    """
    raw_chunks = chunk_text(text, max_tokens=480, overlap=50)
    processed_chunks = []

    for chunk in raw_chunks:
        truncated_chunk = chunk
        while get_token_count(INSTRUCTION, truncated_chunk) > 512:
            tokens = encoding.encode(truncated_chunk)
            if len(tokens) <= 10:
                truncated_chunk = ""
                break
            tokens = tokens[:-10]
            truncated_chunk = encoding.decode(tokens)

        if truncated_chunk:
            processed_chunks.append(truncated_chunk)
        else:
            logger.warning("⚠️ Skipping chunk that exceeded token limit even after truncation.")

    return processed_chunks


async def get_mistral_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Send chunks to Mistral in batches and return embeddings with retries.
    """
    valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    if not valid_chunks:
        raise RuntimeError("No valid chunks to embed after filtering.")

    all_embeddings = []

    logger.debug(f"🔁 Sending {len(valid_chunks)} chunks to Mistral for embeddings in batches of {MAX_BATCH_SIZE}.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(valid_chunks), MAX_BATCH_SIZE):
            batch = valid_chunks[i:i + MAX_BATCH_SIZE]
            success = False

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.debug(f"📤 Sending batch {i // MAX_BATCH_SIZE + 1} (size {len(batch)}) - Attempt {attempt}")
                    payload = {"model": EMBED_MODEL, "input": batch}
                    response = await client.post(MISTRAL_EMBED_URL, headers=HEADERS, json=payload)
                    response.raise_for_status()

                    embeddings_data = response.json().get("data", [])
                    if len(embeddings_data) != len(batch):
                        logger.warning(f"⚠️ API returned {len(embeddings_data)} embeddings for {len(batch)} chunks.")

                    batch_embeddings = [entry["embedding"] for entry in embeddings_data]
                    all_embeddings.extend(batch_embeddings)
                    success = True
                    break  # ✅ Success — no more retries for this batch

                except httpx.HTTPStatusError as e:
                    logger.error(f"🚨 Mistral API Error (Attempt {attempt}): {e.response.text}")
                except Exception as e:
                    logger.exception(f"❌ Unexpected error from Mistral API (Attempt {attempt})")

                # Delay before retrying
                await asyncio.sleep(1.5 * attempt)

            if not success:
                logger.error(f"❌ Failed to embed batch {i // MAX_BATCH_SIZE + 1} after {MAX_RETRIES} retries.")

    return all_embeddings


async def embed_and_store(text: str, collection_name: str) -> int:
    """
    Generate embeddings for text, store in ChromaDB under collection_name.
    """
    chunks = embed_chunks(text)
    if not chunks:
        logger.warning(f"⚠️ No valid chunks generated for collection '{collection_name}'.")
        return 0

    vectors = await get_mistral_embeddings(chunks)

    # ✅ Guard against mismatches
    if len(vectors) != len(chunks):
        logger.warning(f"⚠️ Mismatch: {len(chunks)} chunks but {len(vectors)} embeddings returned. Trimming to match.")
        min_len = min(len(chunks), len(vectors))
        chunks = chunks[:min_len]
        vectors = vectors[:min_len]

    if not chunks or not vectors:
        logger.error(f"❌ No embeddings stored for '{collection_name}'.")
        return 0

    ids = [str(uuid4()) for _ in range(len(chunks))]
    collection = get_or_create_collection(collection_name)
    collection.add(documents=chunks, ids=ids, embeddings=vectors)

    logger.info(f"✅ Stored {len(chunks)} chunks with Mistral embeddings in collection '{collection_name}'")
    return len(chunks)
