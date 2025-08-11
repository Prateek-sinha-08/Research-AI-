# from app.db.chroma_db import get_or_create_collection
# from app.utils.config import settings
# from app.services.embeddings import get_mistral_embeddings
# from mistralai.client import MistralClient
# import asyncio

# mistral = MistralClient(api_key=settings.MISTRAL_API_KEY)

# PROMPT_TEMPLATE = """
# Compare the following research content across papers.

# Context:
# {context}

# ---

# Question:
# Identify the following:
# 1. Novel Contributions
# 2. Similarities across papers
# 3. Research Gaps or Missing Elements

# Format your response as:

# Novel Contributions:
# <your answer>

# Similarities:
# <your answer>

# Missing Gaps:
# <your answer>
# """

# MAX_CONTEXT_CHUNKS = 50      # total max retrieved chunks across all PDFs
# MAX_CHUNK_CHARS = 1500       # trim each chunk to avoid token overflow

# async def query_multi_pdf_collections(collection_names: list[str], question: str, top_k: int = 3) -> str:
#     """
#     Query multiple Chroma collections using precomputed Mistral embeddings.
#     Batch-safe retrieval & context trimming to avoid token errors.
#     """
#     all_results = []

#     # ✅ Get embedding for the question
#     question_embedding = (await get_mistral_embeddings([question]))[0]

#     for name in collection_names:
#         try:
#             collection = get_or_create_collection(name)
#         except Exception as e:
#             raise ValueError(f"Collection '{name}' not found or invalid: {str(e)}")

#         results = collection.query(
#             query_embeddings=[question_embedding],
#             n_results=top_k
#         )

#         documents = results.get('documents', [[]])[0]
#         all_results.extend(documents)

#     # ✅ Deduplicate and trim size
#     unique_context = list(dict.fromkeys(all_results))[:MAX_CONTEXT_CHUNKS]
#     if not unique_context:
#         raise ValueError("No context found from the provided PDF collections.")

#     # ✅ Limit each chunk's size
#     trimmed_context = [(chunk[:MAX_CHUNK_CHARS] + "...") if len(chunk) > MAX_CHUNK_CHARS else chunk
#                        for chunk in unique_context]

#     context = "\n\n---\n\n".join(trimmed_context)

#     # Final prompt
#     prompt = PROMPT_TEMPLATE.format(context=context, question=question)

#     # Send to Mistral safely
#     loop = asyncio.get_event_loop()
#     response = await loop.run_in_executor(
#         None,
#         lambda: mistral.chat(
#             model="mistral-medium",
#             messages=[{"role": "user", "content": prompt}]
#         )
#     )

#     return response.choices[0].message.content


# def parse_rag_output(output: str) -> dict:
#     parsed = {
#         "novel_insights": "",
#         "similarities": "",
#         "missing_gaps": ""
#     }

#     current_section = None
#     for line in output.splitlines():
#         line = line.strip()
#         if line.lower().startswith("novel contributions"):
#             current_section = "novel_insights"
#             continue
#         elif line.lower().startswith("similarities"):
#             current_section = "similarities"
#             continue
#         elif line.lower().startswith("missing gaps"):
#             current_section = "missing_gaps"
#             continue

#         if current_section:
#             parsed[current_section] += line + " "

#     return parsed


from app.db.chroma_db import get_or_create_collection
from app.utils.config import settings
from app.services.embeddings import get_mistral_embeddings
from mistralai.client import MistralClient
import asyncio

mistral = MistralClient(api_key=settings.MISTRAL_API_KEY)

MAX_CONTEXT_CHUNKS = 50      # max total chunks retrieved
MAX_CHUNK_CHARS = 1500       # trim each chunk to avoid token overflow

async def query_multi_pdf_collections(collection_names: list[str], question: str, top_k: int = 3) -> str:
    """
    Query multiple Chroma collections using Mistral embeddings.
    Retrieve context chunks relevant to the question, then ask the model to answer the question using that context.
    """
    all_results = []

    # Get embedding for the question
    question_embedding = (await get_mistral_embeddings([question]))[0]

    # Retrieve relevant chunks from each collection
    for name in collection_names:
        try:
            collection = get_or_create_collection(name)
        except Exception as e:
            raise ValueError(f"Collection '{name}' not found or invalid: {str(e)}")

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        documents = results.get('documents', [[]])[0]
        all_results.extend(documents)

    # Deduplicate and limit total context size
    unique_context = list(dict.fromkeys(all_results))[:MAX_CONTEXT_CHUNKS]
    if not unique_context:
        raise ValueError("No context found from the provided PDF collections.")

    # Trim each chunk to max chars
    trimmed_context = [
        (chunk[:MAX_CHUNK_CHARS] + "...") if len(chunk) > MAX_CHUNK_CHARS else chunk
        for chunk in unique_context
    ]

    # Compose the context string to provide to the model
    context = "\n\n---\n\n".join(trimmed_context)

    # Compose a natural prompt combining context + user question
    prompt = (
        f"You are an AI assistant helping to answer questions based on the following research papers content.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Please provide a detailed and clear answer based on the context above."
    )

    # Send prompt to Mistral chat model safely
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: mistral.chat(
            model="mistral-medium",
            messages=[{"role": "user", "content": prompt}]
        )
    )

    return response.choices[0].message.content
