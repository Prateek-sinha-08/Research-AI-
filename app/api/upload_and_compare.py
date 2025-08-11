from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from typing import List
from uuid import uuid4
from app.utils.sanitizer import sanitize_collection_name, sanitize_text
from app.models.pdf_log import PDFLog
from app.services.pdf_extractor import extract_pdf_data
from app.services.embeddings import embed_and_store
from app.db.chroma_db import get_or_create_collection
from app.models.schemas import ComparisonResult
import logging
import asyncio
from app.services.novelty_detector import get_unique_chunks
from mistralai.client import MistralClient
from app.utils.config import settings

router = APIRouter()
logger = logging.getLogger("uvicorn.error")

mistral = MistralClient(api_key=settings.MISTRAL_API_KEY)

PROMPT_TEMPLATE = """
Compare the following research content for the paper titled "{title}" against other papers titled: {other_titles}.

Context:
{context}

---

Question:
For the paper titled "{title}", please provide:

1. Novel Contributions unique to this paper,
2. Similarities with the other papers,
3. Research Gaps or Missing Elements relevant to this paper.

Format your response exactly as:

Novel Contributions:
<your answer>

Similarities:
<your answer>

Missing Gaps:
<your answer>
"""

MAX_CONTEXT_CHUNKS = 50      # max chunks to include in prompt
MAX_CHUNK_CHARS = 1500      # max chars per chunk to avoid token overflow


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

    # Strip trailing spaces
    for key in parsed:
        parsed[key] = parsed[key].strip()

    return parsed


async def generate_full_rag_summary(title: str, unique_chunks: List[dict], other_titles: List[str]) -> str:
    """
    Build prompt and query Mistral LLM to get full RAG output (novelty, similarity, gaps).
    """
    # Prepare context by joining unique chunks, trimming each chunk
    texts = [
        chunk["text"][:MAX_CHUNK_CHARS] + "..." if len(chunk["text"]) > MAX_CHUNK_CHARS else chunk["text"] 
        for chunk in unique_chunks
    ]

    # Limit total chunks to avoid huge prompt
    texts = texts[:MAX_CONTEXT_CHUNKS]

    context = "\n\n---\n\n".join(texts)

    # Formulate prompt with context and question (with explicit title and other_titles)
    prompt = PROMPT_TEMPLATE.format(title=title, other_titles=", ".join(other_titles), context=context)

    # Run blocking call in executor to not block event loop
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: mistral.chat(
            model="mistral-medium",
            messages=[{"role": "user", "content": prompt}]
        )
    )

    return response.choices[0].message.content


@router.post("/upload-and-compare", response_model=List[ComparisonResult])
async def upload_and_compare(files: List[UploadFile] = File(...)):
    if not (2 <= len(files) <= 5):
        raise HTTPException(status_code=400, detail="Please upload between 2 to 5 PDF files.")

    collection_names = []
    paper_titles = []

    # UPLOAD + EMBEDDING PHASE
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a valid PDF.")

        try:
            existing_log = await PDFLog.filter(filename=file.filename).first()

            if not existing_log:
                contents = await file.read()
                result = await run_in_threadpool(extract_pdf_data, contents, file.filename)

                base_name = file.filename.replace(".pdf", "")
                safe_name = sanitize_collection_name(base_name)
                collection_name = f"{safe_name}_{uuid4().hex[:8]}"

                clean_title = sanitize_text(result["title"])
                clean_excerpt = sanitize_text(result["text_excerpt"])
                clean_full_text = sanitize_text(result["text"])

                await embed_and_store(clean_full_text, collection_name)

                await PDFLog.create(
                    filename=file.filename,
                    title=clean_title,
                    collection_name=collection_name,
                    text_excerpt=clean_excerpt,
                    full_text=clean_full_text,
                )

                collection_names.append(collection_name)
                paper_titles.append(clean_title)

            else:
                collection_names.append(existing_log.collection_name)
                paper_titles.append(existing_log.title)

        except Exception as e:
            logger.exception(f"Failed to process '{file.filename}'.")
            raise HTTPException(status_code=500, detail=f"Failed to process '{file.filename}': {e}")

    # RETRIEVE EMBEDDINGS PER PDF
    all_papers_data = []
    for name in collection_names:
        collection = get_or_create_collection(name)
        paper_data = collection.get(include=["documents", "embeddings"])
        if not paper_data or not paper_data.get("ids"):
            log_entry = await PDFLog.filter(collection_name=name).first()
            if log_entry and log_entry.full_text:
                await embed_and_store(log_entry.full_text, name)
                paper_data = collection.get(include=["documents", "embeddings"])
            else:
                raise HTTPException(status_code=404, detail=f"No stored text for '{name}' to re-embed.")

        combined_data = [{"text": doc, "embedding": emb} for doc, emb in zip(paper_data["documents"], paper_data["embeddings"])]
        all_papers_data.append(combined_data)

    results = []

    # For each PDF, find unique chunks, then generate full RAG output focused on that PDF
    for i in range(len(paper_titles)):
        base_title = paper_titles[i]
        base_chunks = all_papers_data[i]

        others_titles = [paper_titles[j] for j in range(len(paper_titles)) if j != i]
        other_chunks = [chunk for j in range(len(paper_titles)) if j != i for chunk in all_papers_data[j]]

        unique_chunks = await get_unique_chunks(base_chunks, other_chunks)

        rag_output = await generate_full_rag_summary(base_title, unique_chunks, others_titles)
        parsed = parse_rag_output(rag_output)

        # Save back to DB
        log = await PDFLog.filter(title=base_title).first()
        if log:
            log.novel_insights = parsed["novel_insights"]
            log.similarities = parsed["similarities"]
            log.missing_gaps = parsed["missing_gaps"]
            await log.save()

        results.append(ComparisonResult(
            title=base_title,
            novel_insights=[parsed["novel_insights"]],
            similarities=[parsed["similarities"]],
            missing_gaps=[parsed["missing_gaps"]]
        ))

    return results
