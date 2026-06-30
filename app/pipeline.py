"""
Shared extraction pipeline used by the API and background tasks.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from app.classifier import DocumentClassifier
from app.database import db
from app.extractor import ConstrainedExtractor
from app.ocr import OCREngine
from app.validator import ExtractionValidator


class ExtractionPipeline:
    def __init__(
        self,
        ocr_engine: Optional[OCREngine] = None,
        classifier: Optional[DocumentClassifier] = None,
        extractor: Optional[ConstrainedExtractor] = None,
        validator: Optional[ExtractionValidator] = None,
    ):
        self.ocr_engine = ocr_engine or OCREngine()
        self.classifier = classifier or DocumentClassifier()
        self.extractor = extractor or ConstrainedExtractor(
            backend=os.getenv("EXTRACTION_BACKEND", "openai")
        )
        self.validator = validator or ExtractionValidator()

    def process(
        self,
        image_path: str,
        model_version: str = "v0.1.0",
        store: bool = True,
    ) -> Dict[str, Any]:
        raw_text, _ocr_confidence = self.ocr_engine.extract_text_with_confidence(image_path)

        if not raw_text.strip():
            return {
                "success": False,
                "error": "No text extracted from image",
                "status_code": 422,
            }

        doc_type, class_confidence = self.classifier.classify(raw_text, image_path)

        if doc_type == "unknown":
            return {
                "success": False,
                "error": "Could not identify document type",
                "raw_text": raw_text[:500],
                "classification_confidence": class_confidence,
                "status_code": 422,
            }

        raw_extraction = self.extractor.extract(
            raw_text,
            doc_type,
            model_version=model_version,
        )

        validated_doc, metadata = self.validator.validate(
            raw_extraction,
            doc_type,
            raw_confidence_scores=raw_extraction.get("confidence_scores"),
            model_version=model_version,
        )

        extraction_id = None
        if store:
            extraction_id = db.insert_extraction(
                document_type=doc_type,
                image_path=image_path,
                extracted_data=validated_doc.model_dump() if validated_doc else raw_extraction,
                confidence_scores=metadata.get("calibrated_scores"),
                validation_passed=validated_doc is not None,
                status="pending_review" if metadata.get("needs_review") else "completed",
                errors=str(metadata.get("errors")) if metadata.get("errors") else None,
                model_version=model_version,
            )

        response: Dict[str, Any] = {
            "success": True,
            "extraction_id": extraction_id,
            "document_type": doc_type,
            "status": metadata["status"],
            "needs_review": metadata.get("needs_review", False),
        }

        if validated_doc:
            response["extracted_data"] = validated_doc.model_dump()
            if validated_doc.confidence_scores:
                response["confidence_scores"] = validated_doc.confidence_scores
        else:
            response["errors"] = metadata.get("errors", [])

        return response
