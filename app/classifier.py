# app/classifier.py
from typing import Literal

DocumentType = Literal["pan", "indian_passport", "aadhaar", "unknown"]

class DocumentClassifier:
    """
    Classify document type before extraction.
    Could be: regex on text, small vision model, or LLM-based.
    Start simple, swap later.
    """
    
    def classify(self, ocr_text: str, image_path: str) -> tuple[DocumentType, float]:
        """
        Returns (document_type, confidence).
        
        Simple approach: keyword matching + regex patterns.
        Upgrade path: fine-tuned ViT or Qwen2.5-VL for classification.
        """
        text_lower = ocr_text.lower()
        scores = {
            "pan": 0.0,
            "indian_passport": 0.0,
            "aadhaar": 0.0,
        }
        
        # PAN indicators
        if "income tax" in text_lower or "permanent account" in text_lower:
            scores["pan"] += 0.4
        if any(pattern in text_lower for pattern in ["pan", "pan no", "pan number"]):
            scores["pan"] += 0.3
        import re
        if re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', ocr_text):
            scores["pan"] += 0.3
        
        # Passport indicators
        if "passport" in text_lower or "passport no" in text_lower:
            scores["indian_passport"] += 0.4
        if "nationality" in text_lower or "surname" in text_lower:
            scores["indian_passport"] += 0.3
        if re.search(r'\b[A-Z][0-9]{7}\b', ocr_text):
            scores["indian_passport"] += 0.3
        
        # Aadhaar indicators
        if "aadhaar" in text_lower or "aadhar" in text_lower or "uidai" in text_lower:
            scores["aadhaar"] += 0.4
        if "unique identification" in text_lower:
            scores["aadhaar"] += 0.3
        if re.search(r'\b[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\b', ocr_text):
            scores["aadhaar"] += 0.3
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        if confidence < 0.3:
            return "unknown", confidence
        
        return best_type, confidence