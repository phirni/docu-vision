# app/ocr.py
from PIL import Image
from typing import Optional
import pytesseract  # Free, works offline, good enough for printed text

class OCREngine:
    """Thin wrapper. The interesting engineering starts AFTER this."""
    
    def extract_text(self, image_path: str) -> str:
        """Return raw text. That's it. No structure, no intelligence."""
        image = Image.open(image_path)
        # Preprocessing you can tune later if needed:
        # - Convert to grayscale
        # - Threshold for contrast
        # - Deskew
        text = pytesseract.image_to_string(image)
        return text.strip()
    
    def extract_text_with_confidence(self, image_path: str) -> tuple[str, float]:
        """Also return word-level confidence for later calibration."""
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Average word confidence as a rough document quality signal
        confidences = [
            int(conf) for conf in data['conf'] 
            if conf != '-1' and conf != -1
        ]
        avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        
        text = pytesseract.image_to_string(image)
        return text.strip(), avg_confidence