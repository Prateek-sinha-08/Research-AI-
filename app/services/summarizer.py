import asyncio
from mistralai.client import MistralClient
from app.utils.config import settings
from typing import List
from app.utils.text_splitter import chunk_text  # import your better splitter
import logging

logger = logging.getLogger("uvicorn.error")

mistral = MistralClient(api_key=settings.MISTRAL_API_KEY)


async def summarize_chunk(chunk: str, idx: int, total: int) -> str:
    prompt = f"""
You are a research assistant. Summarize the following chunk ({idx} of {total}) of a research paper text into 2-3 paragraphs.
Avoid repeating information from previous chunks.

Paper chunk:
{chunk}
"""
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: mistral.chat(
            model="mistral-medium",
            messages=[{"role": "user", "content": prompt}]
        )
    )
    summary = response.choices[0].message.content.strip()
    logger.debug(f"Chunk {idx}/{total} summary length: {len(summary)} chars")
    return summary


async def summarize_single_paper(paper_text: str) -> str:
    # Use your improved sentence-aware chunker with tokens and overlap
    chunks = chunk_text(paper_text, max_tokens=1500, overlap=150)  
    logger.info(f"Paper split into {len(chunks)} chunks for summarization")

    # Summarize chunks concurrently for speed
    chunk_summaries = await asyncio.gather(
        *[summarize_chunk(chunk, idx + 1, len(chunks)) for idx, chunk in enumerate(chunks)]
    )

    # Aggregate summaries into a final combined summary
    aggregate_prompt = (
        "You are a research assistant. Combine the following partial summaries "
        "of a research paper into a clear, coherent, and concise overall summary "
        "in 3-4 paragraphs. Remove redundancies and synthesize key points.\n\n"
        + "\n\n".join(chunk_summaries)
    )

    loop = asyncio.get_running_loop()
    final_response = await loop.run_in_executor(
        None,
        lambda: mistral.chat(
            model="mistral-medium",
            messages=[{"role": "user", "content": aggregate_prompt}]
        )
    )
    final_summary = final_response.choices[0].message.content.strip()
    logger.info(f"Final summary length: {len(final_summary)} chars")

    return final_summary


async def summarize_papers(paper_titles: List[str], paper_texts: List[str]) -> List[dict]:
    # Summarize all papers concurrently
    tasks = [summarize_single_paper(text) for text in paper_texts]
    results = await asyncio.gather(*tasks)

    return [{"title": title, "summary": summary} for title, summary in zip(paper_titles, results)]
