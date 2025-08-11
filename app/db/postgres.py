from tortoise.contrib.fastapi import register_tortoise
from fastapi import FastAPI
from app.utils.config import settings
import logging

# ✅ Proper logger instance
logger = logging.getLogger("uvicorn.error")

def init_postgres(app: FastAPI):
    """
    Initializes the Tortoise ORM PostgreSQL connection with FastAPI.
    Automatically creates tables for User, Session, PDFLog.
    """
    register_tortoise(
        app,
        db_url=settings.DB_URL,
        modules={"models": ["app.models"]},
        generate_schemas=True,          # Auto create tables (disable in production)
        add_exception_handlers=True,    # Adds HTTP 422 error handlers
    )
    logger.info("✅ PostgreSQL/Tortoise ORM initialized")
