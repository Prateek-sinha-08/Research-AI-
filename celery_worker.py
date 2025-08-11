# celery_worker.py

from app.tasks import celery_app

# Entry point to start Celery worker:
# Run this from the terminal:
# celery -A celery_worker.celery_app worker --loglevel=info
