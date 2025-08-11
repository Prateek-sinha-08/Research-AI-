from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from app.services.rag_pipeline import query_multi_pdf_collections

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str = Field(..., example="What is the key innovation in these papers?")
    collection_names: List[str] = Field(..., example=["paper_0_llm", "paper_1_diffusion"])

@router.post("/ask-question")
async def ask_question(payload: QuestionRequest):
    if not payload.collection_names or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question and collection_names are required.")

    try:
        response = await query_multi_pdf_collections(
            collection_names=payload.collection_names,
            question=payload.question
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

    return {"answer": response}
