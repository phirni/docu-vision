# app/validator.py
"""
Post-extraction validation and confidence calibration.
This is where model output becomes trusted data.
"""

from app.schemas import validate_extraction, BaseIDDocument
from pydantic import ValidationError
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression
import pickle
import os


class ExtractionValidator:
    """
    Validates and calibrates extracted documents.
    
    Two-stage defense:
    1. Schema validation (structure + business rules)
    2. Confidence calibration (is the model honest about uncertainty?)
    """
    
    def __init__(self, calibrator_path: str = "app/calibrator.pkl"):
        self.calibrator = Calibrator.load_or_create(calibrator_path)
    
    def validate(
        self,
        raw_extraction: Dict[str, Any],
        document_type: str,
        raw_confidence_scores: Optional[Dict[str, float]] = None,
        model_version: Optional[str] = None
    ) -> Tuple[Optional[BaseIDDocument], Dict[str, Any]]:
        """
        Validate and calibrate a raw extraction.
        
        Returns:
            (validated_document_or_None, metadata_dict)
        """
        metadata = {
            "status": "pending",
            "errors": [],
            "calibrated_scores": {},
            "needs_review": False
        }
        
        # Stage 1: Schema validation (structural + business rules)
        try:
            validated_doc = validate_extraction(
                raw_extraction, 
                document_type,
                model_version
            )
        except ValidationError as e:
            metadata["status"] = "validation_failed"
            metadata["errors"] = [
                {"field": err["loc"], "message": err["msg"]}
                for err in e.errors()
            ]
            return None, metadata
        
        # Stage 2: Calibrate confidence scores
        if raw_confidence_scores:
            calibrated = self.calibrator.calibrate(raw_confidence_scores)
            validated_doc.confidence_scores = calibrated
            metadata["calibrated_scores"] = calibrated
            
            # Route based on calibrated confidence
            if validated_doc.needs_human_review(threshold=0.8):
                metadata["needs_review"] = True
                metadata["low_confidence_fields"] = (
                    validated_doc.get_low_confidence_fields(threshold=0.8)
                )
        
        metadata["status"] = "validated"
        return validated_doc, metadata


class Calibrator:
    """
    Converts raw model confidence into calibrated probabilities.
    
    Why: LLMs are overconfident. Raw 90% confidence might be 70% accuracy.
    This fixes that using isotonic regression on held-out data.
    """
    
    def __init__(self):
        self.field_calibrators: Dict[str, IsotonicRegression] = {}
        self.fitted = False
    
    @classmethod
    def load_or_create(cls, path: str = "app/calibrator.pkl"):
        """Load existing calibrator or create new one."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return cls()
    
    def fit(self, field_scores: Dict[str, np.ndarray], field_labels: Dict[str, np.ndarray]):
        """Fit isotonic regression per field."""
        for field in field_scores:
            if len(field_scores[field]) < 10:
                continue
            
            cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            cal.fit(field_scores[field], field_labels[field])
            self.field_calibrators[field] = cal
        
        self.fitted = True
    
    def calibrate(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Apply calibration to raw scores."""
        if not self.fitted:
            return raw_scores
        
        calibrated = {}
        for field, score in raw_scores.items():
            if field in self.field_calibrators:
                calibrated[field] = float(
                    self.field_calibrators[field].predict([score])[0]
                )
            else:
                calibrated[field] = score
        
        return calibrated
    
    def measure_ece(
        self, 
        scores: np.ndarray, 
        labels: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """Expected Calibration Error — lower is better."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (scores > bin_boundaries[i]) & (scores <= bin_boundaries[i+1])
            if in_bin.sum() == 0:
                continue
            
            bin_acc = labels[in_bin].mean()
            bin_conf = scores[in_bin].mean()
            ece += (in_bin.sum() / len(scores)) * abs(bin_acc - bin_conf)
        
        return float(ece)
    
    def save(self, path: str = "app/calibrator.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self, f)