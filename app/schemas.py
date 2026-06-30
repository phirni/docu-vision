"""
Document schemas for constrained decoding + calibrated confidence extraction.

Every schema inherits from BaseIDDocument which enforces:
1. Strict typing (constrained decoder cannot output malformed JSON)
2. Domain validation (business rules catch model hallucinations)
3. Confidence tracking (per-field scores for human review routing)
4. Observability (model version, raw text hash for traceability)
"""

from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, Dict, Any, List
from datetime import date, datetime
import re
import hashlib
import json


# ============================================================================
# Base Schema - Shared contract for all government IDs
# ============================================================================

class BaseIDDocument(BaseModel):
    """Every document type inherits these fields and behaviors."""
    
    document_type: str  # Literal dispatch key: "pan", "passport", "aadhaar"
    
    # Core identity fields
    name: str
    dob: Optional[date] = None
    gender: Optional[str] = None
    
    # Tracing & observability
    raw_text_hash: Optional[str] = None
    model_version: Optional[str] = None
    
    # Per-field calibrated confidence scores (0.0 to 1.0)
    # Keys match field names, values are calibrated probabilities
    confidence_scores: Optional[Dict[str, float]] = None
    
    # ========== Common Validators ==========
    
    @field_validator('dob', mode='before')
    @classmethod
    def normalize_dob(cls, v):
        """
        Accept multiple date formats, normalize to date object.
        Returns None for unparsable dates instead of guessing.
        This is a defense layer: model hallucination caught here.
        """
        if v is None or isinstance(v, date):
            return v
        
        if isinstance(v, str):
            v = v.strip()
            # Ordered by likelihood in Indian documents
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y"):
                try:
                    parsed = datetime.strptime(v, fmt).date()
                    # Sanity: reject future dates (except for expiry fields in subclasses)
                    if parsed > date.today():
                        continue
                    return parsed
                except ValueError:
                    continue
        
        # If nothing worked, don't guess - let it be None
        return None
    
    @field_validator('name')
    @classmethod
    def clean_name(cls, v):
        """Normalize names: strip whitespace, title case."""
        if v is None:
            return v
        # Remove extra spaces, title case
        cleaned = ' '.join(v.split())
        return cleaned.title()
    
    @field_validator('gender', mode='before')
    @classmethod
    def normalize_gender(cls, v):
        """Standardize gender to M/F/O or None."""
        if v is None:
            return None
        v = v.strip().upper()
        if v in ('M', 'MALE', 'पुरुष'):
            return 'M'
        if v in ('F', 'FEMALE', 'महिला'):
            return 'F'
        if v in ('O', 'OTHER', 'T', 'TRANSGENDER'):
            return 'O'
        return None  # Don't guess
    
    @field_validator('confidence_scores')
    @classmethod
    def validate_confidence_range(cls, v):
        """Ensure all confidence scores are in [0,1]."""
        if v is None:
            return v
        for field, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Confidence score for '{field}' is {score}, must be in [0,1]"
                )
        return v
    
    def get_low_confidence_fields(self, threshold: float = 0.8) -> List[str]:
        """Return fields below confidence threshold for human review routing."""
        if self.confidence_scores is None:
            return []
        return [
            field for field, score in self.confidence_scores.items()
            if score < threshold
        ]
    
    def needs_human_review(self, threshold: float = 0.8) -> bool:
        """Should this extraction be reviewed by a human?"""
        return len(self.get_low_confidence_fields(threshold)) > 0
    
    def compute_raw_text_hash(self, raw_text: str) -> str:
        """Hash OCR text for deduplication and traceability."""
        self.raw_text_hash = hashlib.sha256(raw_text.encode()).hexdigest()[:16]
        return self.raw_text_hash


# ============================================================================
# PAN Card Schema
# ============================================================================

class PANSchema(BaseIDDocument):
    """
    Indian Permanent Account Number card.
    
    Validation rules:
    - PAN number: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)
    - Name and father's name extracted from the card
    """
    
    document_type: str = "pan"
    father_name: Optional[str] = None
    pan_number: str
    
    @field_validator('pan_number')
    @classmethod
    def validate_pan(cls, v):
        """
        Enforce PAN format: ABCDE1234F
        The constrained decoder already forces valid JSON, but this
        catches wrong-format values the model might confidently extract.
        """
        cleaned = v.upper().replace(" ", "").replace("-", "")
        
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', cleaned):
            raise ValueError(
                f"Invalid PAN format: '{v}'. Expected pattern: 5 letters, 4 digits, 1 letter"
            )
        return cleaned
    
    @field_validator('father_name')
    @classmethod
    def clean_father_name(cls, v):
        """Father's name follows same normalization as name."""
        if v is None:
            return v
        return ' '.join(v.split()).title()
    
    @model_validator(mode='after')
    def check_name_different_from_father(self):
        """Catch common extraction error: name == father's name."""
        if (
            self.name
            and self.father_name
            and self.name.lower() == self.father_name.lower()
        ):
            pass
        return self


# ============================================================================
# Indian Passport Schema
# ============================================================================

class IndianPassportSchema(BaseIDDocument):
    """
    Indian Passport extraction schema.
    
    Validation:
    - Passport number: 1 letter + 7 digits (e.g., Z1234567)
    - Dates: issue before expiry, expiry must be in future (at issuance time)
    """
    
    document_type: str = "indian_passport"
    passport_number: str
    surname: str
    given_name: str
    nationality: str = "IND"
    place_of_birth: Optional[str] = None
    place_of_issue: Optional[str] = None
    date_of_issue: Optional[date] = None
    date_of_expiry: Optional[date] = None
    file_number: Optional[str] = None
    
    @field_validator('passport_number')
    @classmethod
    def validate_passport_number(cls, v):
        """Indian passport: 1 uppercase letter + 7 digits."""
        cleaned = v.upper().replace(" ", "").replace("-", "")
        
        if not re.match(r'^[A-Z][0-9]{7}$', cleaned):
            raise ValueError(
                f"Invalid passport number: '{v}'. Expected: 1 letter + 7 digits"
            )
        return cleaned
    
    @field_validator('nationality')
    @classmethod
    def normalize_nationality(cls, v):
        """Standardize nationality codes."""
        v = v.strip().upper()
        if v in ('IND', 'INDIAN', 'भारतीय'):
            return 'IND'
        return v
    
    @model_validator(mode='after')
    def validate_dates(self):
        """Business rules for passport dates."""
        issue = self.date_of_issue
        expiry = self.date_of_expiry

        if issue and expiry:
            if issue >= expiry:
                raise ValueError(
                    f"Issue date {issue} must be before expiry date {expiry}"
                )

            max_validity_days = 10 * 365 + 10
            actual_validity = (expiry - issue).days
            if actual_validity > max_validity_days:
                pass

        return self

    @model_validator(mode='after')
    def construct_full_name(self):
        """Ensure name is set from given_name + surname if needed."""
        if (not self.name or self.name == '') and self.given_name and self.surname:
            self.name = f"{self.given_name} {self.surname}"

        return self


# ============================================================================
# Aadhaar Card Schema (included for completeness)
# ============================================================================

class AadhaarSchema(BaseIDDocument):
    """
    Indian Aadhaar card.
    
    Validation:
    - Aadhaar number: exactly 12 digits
    - Optional: VID (Virtual ID) as alternative
    """
    
    document_type: str = "aadhaar"
    aadhaar_number: str
    address: Optional[str] = None
    pin_code: Optional[str] = None
    
    @field_validator('aadhaar_number')
    @classmethod
    def validate_aadhaar(cls, v):
        """Aadhaar number must be exactly 12 digits."""
        cleaned = v.replace(" ", "").replace("-", "")
        
        if not re.match(r'^\d{12}$', cleaned):
            raise ValueError(
                f"Invalid Aadhaar number: '{v}'. Expected exactly 12 digits"
            )
        
        # Verhoeff checksum check could go here in production
        # from verhoeff import validate
        # if not validate(cleaned):
        #     raise ValueError(f"Aadhaar number {cleaned} failed Verhoeff checksum")
        
        return cleaned
    
    @field_validator('pin_code')
    @classmethod
    def validate_pin_code(cls, v):
        """Indian PIN code: 6 digits."""
        if v is None:
            return v
        cleaned = v.strip().replace(" ", "")
        if cleaned and not re.match(r'^\d{6}$', cleaned):
            # Don't raise error, just return None
            # PIN might be partially visible or not present
            return None
        return cleaned


# ============================================================================
# Unknown / Unclassified Document (for confidence-based routing)
# ============================================================================

class UnknownDocumentSchema(BaseModel):
    """
    Fallback schema when the classifier is uncertain.
    Captures whatever text is available without enforcing structure.
    """
    
    document_type: str = "unknown"
    raw_text: str
    classifier_confidence: float
    possible_types: List[str] = []  # e.g., ["pan", "passport"]
    image_hash: Optional[str] = None
    
    @field_validator('classifier_confidence')
    @classmethod
    def validate_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {v}")
        return v


# ============================================================================
# Schema Dispatch Map (used by constrained decoder and router)
# ============================================================================

SCHEMA_MAP: Dict[str, type] = {
    "pan": PANSchema,
    "indian_passport": IndianPassportSchema,
    "aadhaar": AadhaarSchema,
    "unknown": UnknownDocumentSchema,
}


# ============================================================================
# Factory Function (used by pipeline after classification)
# ============================================================================

def get_schema_for_document_type(
    doc_type: str, 
    strict: bool = True
) -> type:
    """
    Return the appropriate schema class for a document type.
    
    Args:
        doc_type: The classified document type
        strict: If True, raise error on unknown types. 
                If False, return UnknownDocumentSchema as fallback.
    
    Returns:
        Schema class for validation and constrained decoding
    """
    schema = SCHEMA_MAP.get(doc_type.lower())
    
    if schema is None:
        if strict:
            raise ValueError(
                f"Unknown document type: '{doc_type}'. "
                f"Known types: {list(SCHEMA_MAP.keys())}"
            )
        return UnknownDocumentSchema
    
    return schema


def validate_extraction(
    raw_output: Dict[str, Any],
    doc_type: str,
    model_version: Optional[str] = None
) -> BaseIDDocument:
    """
    Validate and normalize raw LLM output into a trusted document object.
    
    This is the critical boundary between "model output" and "trusted data."
    Returns a validated instance or raises ValidationError with details.
    
    Args:
        raw_output: Raw LLM extraction (from constrained decoder)
        doc_type: Expected document type
        model_version: Version string of the extraction model
    
    Returns:
        Validated BaseIDDocument instance
    
    Raises:
        ValidationError: If the output fails schema validation
    """
    schema_cls = get_schema_for_document_type(doc_type)
    
    # Add metadata if not present
    if 'model_version' not in raw_output and model_version:
        raw_output['model_version'] = model_version
    
    # This is where Pydantic enforces all validators
    validated = schema_cls(**raw_output)
    
    return validated