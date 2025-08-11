import tiktoken
from nltk.tokenize import sent_tokenize
from typing import List
import logging

# Setup logger
logger = logging.getLogger("uvicorn.error")

# OpenAI tokenizer (compatible with Mistral's embedding limits)
encoding = tiktoken.get_encoding("cl100k_base")

def get_token_count(text: str) -> int:
    return len(encoding.encode(text))

def chunk_text(text: str, max_tokens: int = 2048, overlap: int = 100) -> List[str]:
    """
    Splits text into token-limited chunks using sentence-based splitting and overlap.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = get_token_count(sentence)

        if current_len + sentence_len > max_tokens:
            chunks.append(" ".join(current_chunk))

            # Apply overlap logic
            if overlap > 0:
                overlap_sentences = []
                total = 0
                for s in reversed(current_chunk):
                    slen = get_token_count(s)
                    if total + slen <= overlap:
                        overlap_sentences.insert(0, s)
                        total += slen
                    else:
                        break
                current_chunk = overlap_sentences
                current_len = total
            else:
                current_chunk = []
                current_len = 0

        current_chunk.append(sentence)
        current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.debug(f"📚 Text split into {len(chunks)} chunks (max_tokens={max_tokens}, overlap={overlap})")

    return chunks
