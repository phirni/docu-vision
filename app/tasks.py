"""
Background task definitions for async processing.
Use if you want to process documents in the background instead of synchronously.
"""

import os

from celery import Celery

from app.pipeline import ExtractionPipeline

celery_app = Celery(
    "extraction_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
)


@celery_app.task
def process_document_async(image_path: str, model_version: str = "v0.1.0"):
    """Process a document asynchronously."""
    pipeline = ExtractionPipeline()
    return pipeline.process(image_path, model_version=model_version)
