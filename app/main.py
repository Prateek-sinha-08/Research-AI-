from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.api.upload_and_compare import router as upload_and_compare_router
from app.api.generate import router as generate_router
from app.api.list_files import router as list_files_router
from app.api.ask_question import router as ask_router
from app.services import rag_compare
from app.utils.config import settings
from app.db.postgres import init_postgres

# 🌟 Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="AutoResearch AI",
    description="An intelligent research automation backend for researchers like you",
    version="1.0.0"
)

# ✅ Init Postgres
logger.info("🚀 Calling init_postgres(app) to initialize DB...")
init_postgres(app)
logger.info("✅ Finished calling init_postgres(app)")

# 🧩 Register routers
app.include_router(upload_and_compare_router, prefix="/api")
app.include_router(generate_router, prefix="/api")
# app.include_router(rag_compare.router, prefix="/api")
app.include_router(ask_router, prefix="/api")
app.include_router(list_files_router, prefix="/api")

# 🌐 CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 🚀 Log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"➡️ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"⬅️ {request.method} {request.url.path} - Status: {response.status_code}")
    return response

@app.get("/")
async def root():
    return {"message": "Welcome to AutoResearch AI backend"}
