from fastapi import APIRouter
from app.models.pdf_log import PDFLog
from typing import List 

router = APIRouter()

@router.get("/list-pdfs")
async def list_uploaded_pdfs():
    pdfs=await PDFLog.all().order_by("-id")
    return[
        {
            "filename":p.filename,
            "title":p.title,
            "collection_name":p.collection_name,
            "uploaded_at":p.id
        }
        for p in pdfs
    ]
@router.get("/list-uploaded-collections", response_model=List[str])
async def list_uploaded_collections():
    collections = await PDFLog.all().distinct().values_list("collection_name", flat=True)
    return collections