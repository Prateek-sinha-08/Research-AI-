# app/tasks.py

from celery import Celery
from app.utils.config import settings

# Initialize Celery with Redis broker URL from .env
celery_app = Celery(
    "autoresearch_tasks",
    broker=settings.REDIS_BROKER_URL,
)

# Example: Background task to embed PDF (replace with real logic)
@celery_app.task
def embed_pdf(pdf_id: int):
    print(f"[CELERY] Embedding started for PDF ID: {pdf_id}")
    # Simulate time-consuming logic (like loading models, generating embeddings)
    # You can import and call your actual PDF logic here
    return {"status": "completed", "pdf_id": pdf_id}
