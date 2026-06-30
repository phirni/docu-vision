"""
FastAPI application for document extraction pipeline.
"""

from dotenv import load_dotenv

load_dotenv()

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.database import db
from app.pipeline import ExtractionPipeline

app = FastAPI(title="Document Extraction Pipeline")
pipeline = ExtractionPipeline()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    """
    Upload a document image and extract structured data.

    Flow: Upload → OCR → Classify → Extract → Validate → Store → Return
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

    image_path = UPLOAD_DIR / file.filename
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = pipeline.process(str(image_path))

        if not result.get("success"):
            status_code = result.pop("status_code", 422)
            return JSONResponse(status_code=status_code, content=result)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/review")
def get_pending_reviews():
    """Get extractions that need human review."""
    pending = db.get_pending_reviews()
    return {"count": len(pending), "extractions": pending}


@app.post("/review/{extraction_id}")
def submit_review(
    extraction_id: int,
    status: str = Query(..., pattern="^(approved|rejected)$"),
    notes: str | None = None,
):
    """Submit human review decision."""
    db.update_status(extraction_id, status, notes)
    return {"extraction_id": extraction_id, "status": status}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
