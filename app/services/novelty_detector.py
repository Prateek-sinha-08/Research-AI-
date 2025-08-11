# # # app/services/novelty_detector.py
# # from app.utils.config import settings
# # from mistralai.client import MistralClient
# # from app.services.embeddings import get_mistral_embeddings  # async
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np

# # # Mistral client for summarization
# # mistral = MistralClient(api_key=settings.MISTRAL_API_KEY)


# # async def get_unique_chunks(base_chunks, comparison_chunks, threshold=0.75):
# #     """
# #     Compare base paper chunks with comparison paper chunks and return unique ones.
# #     Uses Mistral embeddings to measure similarity.
# #     """
# #     base_texts = [c["text"] for c in base_chunks]
# #     comp_texts = [c["text"] for c in comparison_chunks]

# #     # Get embeddings using async mistral API call
# #     base_vecs = np.array(await get_mistral_embeddings(base_texts))
# #     comp_vecs = np.array(await get_mistral_embeddings(comp_texts))

# #     # Compute cosine similarities between base and comparison vectors
# #     similarities = cosine_similarity(base_vecs, comp_vecs)

# #     # Keep only those base chunks whose max similarity is below the threshold
# #     unique_indices = [i for i, row in enumerate(similarities) if max(row) < threshold]
# #     return [base_chunks[i]["text"] for i in unique_indices]


# # def generate_novelty_summary(title, unique_chunks, all_titles):
# #     """
# #     Ask Mistral to summarize the novel contributions of a paper
# #     based on its unique chunks.
# #     """
# #     if not unique_chunks:
# #         return "No novel content detected for this paper."

# #     # Use only the first few chunks for summarization to keep prompt size small
# #     context = "\n\n".join(unique_chunks[:5])
# #     prompt = f"""
# #     The following content is found only in the paper titled '{title}' and not in others: {', '.join(all_titles)}.

# #     --- UNIQUE CONTEXT START ---
# #     {context}
# #     --- UNIQUE CONTEXT END ---

# #     Please summarize the novel contributions, unexplored ideas, or distinctive insights from this content.
# #     """

# #     response = mistral.chat(
# #         model="mistral-medium",
# #         messages=[{"role": "user", "content": prompt}]
# #     )

# #     return response.choices[0].message.content

# # # def batch_list(lst, batch_size):
# # #     """Yield successive batch_size-sized chunks from lst."""
# # #     for i in range(0, len(lst), batch_size):
# # #         yield lst[i:i + batch_size]


# app/services/novelty_detector.py
from app.utils.config import settings
from mistralai.client import MistralClient
from app.services.embeddings import get_mistral_embeddings  # async
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio
import logging

logger = logging.getLogger("uvicorn.error")
mistral = MistralClient(api_key=settings.MISTRAL_API_KEY)

async def get_unique_chunks(base_chunks, comparison_chunks, threshold=0.60):
    """
    Compare base paper chunks with comparison paper chunks and return unique ones.
    Uses Mistral embeddings to measure similarity.
    """
    base_texts = [c["text"] for c in base_chunks]
    comp_texts = [c["text"] for c in comparison_chunks]

    if not base_texts or not comp_texts:
        logger.warning("Empty chunks passed to get_unique_chunks")
        return base_texts

    # Batch embedding calls for all base and comparison chunks
    base_vecs = np.array(await get_mistral_embeddings(base_texts))
    comp_vecs = np.array(await get_mistral_embeddings(comp_texts))

    # Compute cosine similarities between base and comparison vectors
    similarities = cosine_similarity(base_vecs, comp_vecs)

    # Keep only those base chunks whose max similarity is below the threshold
    unique_indices = [i for i, row in enumerate(similarities) if max(row) < threshold]
    unique_texts = [base_chunks[i]["text"] for i in unique_indices]

    logger.info(f"Identified {len(unique_texts)} unique chunks out of {len(base_chunks)} base chunks")
    return unique_texts


async def generate_novelty_summary(title, unique_chunks, all_titles):
    """
    Ask Mistral to summarize the novel contributions of a paper
    based on its unique chunks asynchronously.
    """
    if not unique_chunks:
        return "No novel content detected for this paper."

    context = "\n\n".join(unique_chunks[:5])  # limit chunk count to control prompt size

    prompt = f"""
The following content is found only in the paper titled '{title}' and not in others: {', '.join(all_titles)}.

--- UNIQUE CONTEXT START ---
{context}
--- UNIQUE CONTEXT END ---

Please summarize the novel contributions, unexplored ideas, or distinctive insights from this content.
"""

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: mistral.chat(
            model="mistral-medium",
            messages=[{"role": "user", "content": prompt}]
        )
    )

    return response.choices[0].message.content

