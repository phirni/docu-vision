"""
Constrained extraction engine using Groq API (free tier).
Fast, reliable, no local GPU needed.
"""

from typing import Type, Dict, Any, Optional
from pydantic import BaseModel
from app.schemas import SCHEMA_MAP, BaseIDDocument
import json
import os
from groq import Groq


class ConstrainedExtractor:
    """
    Extracts structured data using Groq's API with JSON mode.
    
    Free tier: 30 requests/minute, enough for development.
    Uses structured output to force valid JSON matching your schema.
    """
    
    def __init__(self, model: str = None):
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def extract(
        self,
        ocr_text: str,
        document_type: str,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract structured data using Groq with JSON mode."""
        
        schema = SCHEMA_MAP.get(document_type)
        if schema is None:
            raise ValueError(f"No schema for document type: {document_type}")
        
        # Get schema as JSON for the prompt
        schema_json = schema.model_json_schema()
        field_descriptions = self._format_schema_for_prompt(schema_json)
        
        system_prompt = f"""You extract information from OCR text into JSON.
Return ONLY a valid JSON object. No other text.
If a field is not visible, set it to null.

The JSON must have these fields:
{field_descriptions}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"OCR Text:\n{ocr_text}\n\nExtract JSON:"}
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},  # Forces valid JSON
        )
        
        content = response.choices[0].message.content
        extracted = json.loads(content)
        
        # Add metadata
        extracted["document_type"] = document_type
        if model_version:
            extracted["model_version"] = model_version
        
        return extracted
    
    def extract_with_confidence(
        self,
        ocr_text: str,
        document_type: str,
        model_version: Optional[str] = None
    ) -> tuple[Dict, Dict[str, float]]:
        """Extract and estimate per-field confidence."""
        
        result = self.extract(ocr_text, document_type, model_version)
        
        # Ask model to rate its confidence
        fields = {k: v for k, v in result.items() 
                  if v is not None and k not in ("document_type", "model_version", "confidence_scores")}
        
        if not fields:
            return result, {}
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Rate your confidence (0.0-1.0) in each extracted field. Return JSON: {\"field_name\": 0.95, ...}"},
                {"role": "user", "content": f"OCR: {ocr_text[:300]}\nExtracted: {json.dumps(fields)}\n\nConfidence scores:"}
            ],
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        
        try:
            scores = json.loads(response.choices[0].message.content)
            scores = {k: max(0.0, min(1.0, float(v))) for k, v in scores.items()}
        except (json.JSONDecodeError, ValueError):
            scores = {}
        
        result["confidence_scores"] = scores
        return result, scores
    
    def _format_schema_for_prompt(self, schema_json: dict) -> str:
        """Convert JSON schema to readable field list for prompt."""
        props = schema_json.get("properties", {})
        required = schema_json.get("required", [])
        
        lines = []
        for field, details in props.items():
            field_type = details.get("type", "string")
            req = " (required)" if field in required else " (optional)"
            desc = details.get("description", "")
            lines.append(f"  - {field}: {field_type}{req} {desc}")
        
        return "\n".join(lines)


def create_extractor(model: Optional[str] = None) -> ConstrainedExtractor:
    return ConstrainedExtractor(model=model)