# app/api/generate.py

from fastapi import APIRouter, HTTPException
from typing import List
from app.services.summarizer import summarize_papers
from app.models.pdf_log import PDFLog
from app.models.schemas import SimpleSummary  # ✅ Use from your schema

router = APIRouter()

@router.post("/generate-summary/latest", response_model=List[SimpleSummary])
async def generate_summary_latest_pdfs(limit: int = 2):
    records = await PDFLog.all().order_by("-uploaded_at").limit(limit)

    if not records:
        raise HTTPException(status_code=404, detail="No uploaded PDFs found.")

    paper_titles = [record.title for record in records]
    paper_texts = [getattr(record, "full_text", record.full_text) for record in records]

    summaries = await summarize_papers(paper_titles, paper_texts)

    # Ensure they match the structure expected by SimpleSummary
    return [SimpleSummary(**summary) for summary in summaries]
