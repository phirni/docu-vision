# app/extractor.py
"""
Constrained extraction engine.
Forces LLM output to match Pydantic schemas.
"""

from typing import Type, Dict, Any, Optional
from pydantic import BaseModel
from app.schemas import SCHEMA_MAP, BaseIDDocument
import json
import os


class ConstrainedExtractor:
    """
    Extracts structured data from OCR text using constrained generation.
    
    The Pydantic schema becomes the grammar — the model cannot output
    invalid JSON. This is the boundary between "model guess" and "validated data."
    
    Supports multiple backends:
    - openai: Native structured outputs (fastest path, API required)
    - outlines: Local constrained generation (token masking, more impressive)
    - vllm: If you're using vLLM's guided decoding
    """
    
    def __init__(self, backend: str = "openai"):
        self.backend = backend
        
        # Lazy load model clients
        self._openai_client = None
        self._outlines_model = None
    
    def extract(
        self,
        ocr_text: str,
        document_type: str,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data constrained to the document's schema.
        
        Args:
            ocr_text: Raw OCR output from the image
            document_type: "pan", "indian_passport", "aadhaar"
            model_version: Version tag for observability
        
        Returns:
            Dictionary guaranteed to parse into the target schema
        """
        # Get the right schema for this document type
        schema = SCHEMA_MAP.get(document_type)
        if schema is None:
            raise ValueError(f"No schema for document type: {document_type}")
        
        # Dispatch to backend
        if self.backend == "openai":
            result = self._extract_openai(ocr_text, schema)
        elif self.backend == "outlines":
            result = self._extract_outlines(ocr_text, schema)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        # Add metadata
        result["document_type"] = document_type
        if model_version:
            result["model_version"] = model_version
        
        return result
    
    # ========== OpenAI Backend ==========
    
    def _extract_openai(
        self, 
        ocr_text: str, 
        schema: Type[BaseModel]
    ) -> Dict:
        """OpenAI structured outputs — simplest integration."""
        from openai import OpenAI
        
        if self._openai_client is None:
            self._openai_client = OpenAI()
        
        response = self._openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract information from the OCR text into the required JSON structure. "
                        "If a field is not clearly visible in the text, set it to null. "
                        "Do not hallucinate or guess values."
                    )
                },
                {"role": "user", "content": ocr_text}
            ],
            response_format=schema,  # ← The Pydantic model IS the constraint
            temperature=0.0,  # Deterministic for extraction
        )
        
        # Parse the guaranteed-valid JSON
        return json.loads(response.choices[0].message.content)
    
    # ========== Outlines Backend (Your Depth Area) ==========
    
    def _extract_outlines(
        self, 
        ocr_text: str, 
        schema: Type[BaseModel]
    ) -> Dict:
        """
        Local constrained generation with Outlines.
        
        This is where the real engineering happens:
        - Schema → regex grammar compilation
        - Token-by-token masking during sampling
        - Zero chance of malformed output
        """
        import outlines
        from outlines import models, generate
        
        # Load model once (cached after first call)
        if self._outlines_model is None:
            model_name = os.getenv(
                "LOCAL_MODEL", 
                "Qwen/Qwen2.5-1.5B-Instruct"
            )
            self._outlines_model = models.transformers(
                model_name,
                device="cuda" if self._has_gpu() else "cpu"
            )
        
        # Compile schema → token constraint grammar
        # This is the key step: Pydantic model becomes a regex that
        # filters the token vocabulary at each generation step
        generator = generate.json(
            self._outlines_model, 
            schema,
            sampler=outlines.samplers.greedy()  # Deterministic for extraction
        )
        
        prompt = f"""Extract information from this OCR text into valid JSON.
Only include fields that are clearly visible. Return null for missing fields.

OCR Text:
{ocr_text}

JSON Output:"""
        
        # Generate with token masking active
        # At each step, tokens that would break the JSON are masked to -inf
        result = generator(prompt)
        
        # result is already a Pydantic model instance
        return result.model_dump()
    
    # ========== Utilities ==========
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available for local models."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def extract_with_logprobs(
        self,
        ocr_text: str,
        document_type: str
    ) -> tuple[Dict, Dict[str, float]]:
        """
        Extract AND return token-level probabilities for calibration.
        
        This is needed for your second depth area (calibrated confidence).
        """
        # Extract normally
        result = self.extract(ocr_text, document_type)
        
        # Collect logprobs from the model
        # (Implementation depends on backend — OpenAI returns logprobs,
        #  Outlines can be configured to return them)
        raw_scores = self._collect_token_scores(ocr_text, document_type)
        
        return result, raw_scores
    
    def _collect_token_scores(
        self, 
        ocr_text: str, 
        document_type: str
    ) -> Dict[str, float]:
        """
        Collect raw token probabilities per field.
        These will be calibrated later.
        """
        if self.backend == "openai":
            return self._collect_openai_logprobs(ocr_text, document_type)
        # Outlines logprob collection needs model-specific implementation
        # For now, return empty — you build this when you do calibration
        return {}
    
    def _collect_openai_logprobs(
        self, 
        ocr_text: str, 
        document_type: str
    ) -> Dict[str, float]:
        """Get token-level logprobs from OpenAI."""
        from openai import OpenAI
        
        schema = SCHEMA_MAP[document_type]
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract information into JSON."},
                {"role": "user", "content": ocr_text}
            ],
            response_format={"type": "json_object"},
            logprobs=True,
            top_logprobs=5,
            temperature=0.0,
        )
        
        # Aggregate logprobs per field
        # This is non-trivial — you need to map tokens to fields
        # Simplified version for now
        raw_scores = {}
        if response.choices[0].logprobs:
            tokens = response.choices[0].logprobs.content
            probs = [np.exp(t.logprob) for t in tokens if t.logprob]
            if probs:
                raw_scores["mean_token_confidence"] = float(np.mean(probs))
                raw_scores["min_token_confidence"] = float(min(probs))
        
        return raw_scores


# ========== Factory Function ==========

def create_extractor(backend: Optional[str] = None) -> ConstrainedExtractor:
    """
    Create extractor based on environment configuration.
    Falls back to OpenAI if no config provided.
    """
    if backend is None:
        backend = os.getenv("EXTRACTION_BACKEND", "openai")
    return ConstrainedExtractor(backend=backend)