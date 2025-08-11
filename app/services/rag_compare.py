# # app/api/rag_compare.py

# from app.services.rag_pipeline import query_multi_pdf_collections, parse_rag_output
# from fastapi import APIRouter, HTTPException
# from typing import List
# from app.models.schemas import CompareByFilenamesRequest, ComparisonResult
# from app.models.pdf_log import PDFLog

# router = APIRouter()

# MAX_BATCH_SIZE = 64  # Avoid sending too many chunks to Mistral in one go

# @router.get("/list-uploaded-pdfs", response_model=List[str])
# async def list_uploaded_pdfs():
#     """
#     Returns a list of all uploaded PDF filenames from the database.
#     """
#     files = await PDFLog.all().values_list("filename", flat=True)
#     return list(files)

# @router.post("/rag-compare-pdfs", response_model=List[ComparisonResult])
# async def rag_compare_pdfs(data: CompareByFilenamesRequest):
#     filenames = data.filenames

#     if len(filenames) < 2:
#         raise HTTPException(status_code=400, detail="Please provide at least 2 PDF filenames.")
#     if len(filenames) > 5:
#         raise HTTPException(status_code=400, detail="You can only compare up to 5 PDF filenames.")

#     paper_titles = []
#     collection_names = []

#     for name in filenames:
#         entry = await PDFLog.filter(filename=name).first()
#         if not entry:
#             raise HTTPException(status_code=404, detail=f"No PDF found with filename '{name}'")
#         paper_titles.append(entry.title)
#         collection_names.append(entry.collection_name)

#     question = "Identify the novel contributions, similarities, and research gaps across these papers."

#     try:
#         # Batch-safe multi-PDF query
#         rag_response = await query_multi_pdf_collections(collection_names, question, top_k=MAX_BATCH_SIZE)
#         parsed = parse_rag_output(rag_response)

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

#     result = ComparisonResult(
#         title="Multi-PDF RAG Comparison",
#         novel_insights=[parsed["novel_insights"]],
#         similarities=[parsed["similarities"]],
#         missing_gaps=[parsed["missing_gaps"]]
#     )

#     return [result]


from app.db.chroma_db import get_or_create_collection
from app.utils.config import settings
from app.services.embeddings import get_mistral_embeddings
from mistralai.client import MistralClient
import asyncio

mistral = MistralClient(api_key=settings.MISTRAL_API_KEY)

PROMPT_TEMPLATE = """
Compare the following research content across papers.

Context:
{context}

---

Question:
Identify the following:
1. Novel Contributions
2. Similarities across papers
3. Research Gaps or Missing Elements

Format your response as:

Novel Contributions:
<your answer>

Similarities:
<your answer>

Missing Gaps:
<your answer>
"""

MAX_CONTEXT_CHUNKS = 50      # total max retrieved chunks across all PDFs
MAX_CHUNK_CHARS = 1500       # trim each chunk to avoid token overflow

async def query_multi_pdf_collections(collection_names: list[str], question: str, top_k: int = 3) -> str:
    """
    Query multiple Chroma collections using precomputed Mistral embeddings.
    Batch-safe retrieval & context trimming to avoid token errors.
    """
    all_results = []

    # ✅ Get embedding for the question
    question_embedding = (await get_mistral_embeddings([question]))[0]

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

    # ✅ Deduplicate and trim size
    unique_context = list(dict.fromkeys(all_results))[:MAX_CONTEXT_CHUNKS]
    if not unique_context:
        raise ValueError("No context found from the provided PDF collections.")

    # ✅ Limit each chunk's size
    trimmed_context = [(chunk[:MAX_CHUNK_CHARS] + "...") if len(chunk) > MAX_CHUNK_CHARS else chunk
                       for chunk in unique_context]

    context = "\n\n---\n\n".join(trimmed_context)

    # Final prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # Send to Mistral safely
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: mistral.chat(
            model="mistral-medium",
            messages=[{"role": "user", "content": prompt}]
        )
    )

    return response.choices[0].message.content


def parse_rag_output(output: str) -> dict:
    parsed = {
        "novel_insights": "",
        "similarities": "",
        "missing_gaps": ""
    }

    current_section = None
    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith("novel contributions"):
            current_section = "novel_insights"
            continue
        elif line.lower().startswith("similarities"):
            current_section = "similarities"
            continue
        elif line.lower().startswith("missing gaps"):
            current_section = "missing_gaps"
            continue

        if current_section:
            parsed[current_section] += line + " "

    return parsed
