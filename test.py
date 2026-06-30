"""
Quick pipeline test — run this before starting the server.
Checks each component independently and reports what works/fails.
"""

import sys
from pathlib import Path

# Add your project to path if needed
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("DOCUMENT EXTRACTION PIPELINE — COMPONENT TEST")
print("=" * 60)

# ============================================================
# Test 1: Schemas import and validate correctly
# ============================================================
print("\n[1/6] Testing schemas...")
try:
    from app.schemas import (
        PANSchema,
        IndianPassportSchema,
        AadhaarSchema,
        get_schema_for_document_type,
    )
    
    # Test PAN schema with valid data
    valid_pan = {
        "document_type": "pan",
        "name": "Arjun Mehta",
        "father_name": "Rajesh Mehta",
        "pan_number": "ABCDE1234F",
        "dob": "15/08/1990",
        "gender": "M",
        "confidence_scores": {"pan_number": 0.98, "name": 0.95}
    }
    pan_doc = PANSchema(**valid_pan)
    assert pan_doc.pan_number == "ABCDE1234F", "PAN validation failed"
    assert pan_doc.dob.strftime("%Y-%m-%d") == "1990-08-15", "Date normalization failed"
    
    # Test PAN with invalid number (should raise error)
    try:
        PANSchema(**{**valid_pan, "pan_number": "INVALID"})
        print("  ❌ Should have rejected invalid PAN number")
    except Exception:
        pass  # Expected behavior
    
    # Test schema dispatch
    assert get_schema_for_document_type("pan") == PANSchema
    assert get_schema_for_document_type("indian_passport") == IndianPassportSchema
    
    print("  ✅ Schemas: Import, validation, and dispatch work correctly")
    
except Exception as e:
    print(f"  ❌ Schema test failed: {e}")
    sys.exit(1)

# ============================================================
# Test 2: OCR engine works
# ============================================================
print("\n[2/6] Testing OCR engine...")
try:
    from app.ocr import OCREngine
    
    ocr = OCREngine()
    
    # Create a simple test image with text
    from PIL import Image, ImageDraw, ImageFont
    test_img = Image.new('RGB', (600, 200), color='white')
    draw = ImageDraw.Draw(test_img)
    
    # Try to use a basic font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((20, 20), "INCOME TAX DEPARTMENT", fill='black', font=font)
    draw.text((20, 50), "PAN Card", fill='black', font=font)
    draw.text((20, 80), "Name: ARJUN MEHTA", fill='black', font=font)
    draw.text((20, 110), "PAN: ABCDE1234F", fill='black', font=font)
    
    test_img_path = "test_image.png"
    test_img.save(test_img_path)
    
    # Test text extraction
    text, confidence = ocr.extract_text_with_confidence(test_img_path)
    
    assert len(text) > 0, "OCR returned empty text"
    assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
    
    print(f"  ✅ OCR works — extracted {len(text)} chars, confidence: {confidence:.2f}")
    print(f"     Sample text: {text[:100]}...")
    
    # Cleanup test image
    Path(test_img_path).unlink()
    
except ImportError as e:
    print(f"  ⚠️  OCR import failed (missing dependency?): {e}")
    print("     Install with: pip install pytesseract pillow")
    print("     Also install Tesseract: brew install tesseract (Mac) or apt install tesseract-ocr (Linux)")
except Exception as e:
    print(f"  ❌ OCR test failed: {e}")

# ============================================================
# Test 3: Document classifier works
# ============================================================
print("\n[3/6] Testing classifier...")
try:
    from app.classifier import DocumentClassifier
    
    classifier = DocumentClassifier()
    
    # Test with PAN-like text
    pan_text = """
    INCOME TAX DEPARTMENT
    GOVT OF INDIA
    Permanent Account Number Card
    PAN: ABCDE1234F
    Name: ARJUN MEHTA
    Father's Name: RAJESH MEHTA
    DOB: 15/08/1990
    """
    doc_type, confidence = classifier.classify(pan_text, "dummy.jpg")
    assert doc_type == "pan", f"Should classify as pan, got {doc_type}"
    assert confidence > 0.3, f"Confidence too low: {confidence}"
    print(f"  ✅ Classifier: PAN text → '{doc_type}' (confidence: {confidence:.2f})")
    
    # Test with passport-like text
    passport_text = """
    PASSPORT
    REPUBLIC OF INDIA
    Passport No: Z1234567
    Surname: MEHTA
    Given Name: ARJUN
    Nationality: INDIAN
    """
    doc_type, confidence = classifier.classify(passport_text, "dummy.jpg")
    assert doc_type == "indian_passport", f"Should classify as passport, got {doc_type}"
    print(f"  ✅ Classifier: Passport text → '{doc_type}' (confidence: {confidence:.2f})")
    
    # Test with unknown text
    unknown_text = "This is just random garbage text with no document indicators."
    doc_type, confidence = classifier.classify(unknown_text, "dummy.jpg")
    print(f"  ✅ Classifier: Unknown text → '{doc_type}' (confidence: {confidence:.2f})")
    
except Exception as e:
    print(f"  ❌ Classifier test failed: {e}")

# ============================================================
# Test 4: Validator works
# ============================================================
print("\n[4/6] Testing validator...")
try:
    from app.validator import ExtractionValidator, Calibrator
    
    validator = ExtractionValidator()
    
    # Test validation with good data
    valid_pan = {
        "document_type": "pan",
        "name": "Arjun Mehta",
        "father_name": "Rajesh Mehta",
        "pan_number": "ABCDE1234F",
        "dob": "15/08/1990",
        "gender": "M"
    }
    
    validated_doc, metadata = validator.validate(
        valid_pan, 
        document_type="pan",
        model_version="test"
    )
    
    assert validated_doc is not None, "Valid document should pass validation"
    assert metadata["status"] == "validated"
    print(f"  ✅ Validator: Valid PAN passes validation")
    
    # Test validation with bad data
    bad_pan = {**valid_pan, "pan_number": "INVALID123"}
    validated_doc, metadata = validator.validate(
        bad_pan,
        document_type="pan",
        model_version="test"
    )
    
    assert validated_doc is None, "Invalid PAN should fail validation"
    assert metadata["status"] == "validation_failed"
    assert len(metadata["errors"]) > 0
    print(f"  ✅ Validator: Invalid PAN correctly rejected")
    print(f"     Error: {metadata['errors'][0]['message']}")
    
    # Test calibrator
    import numpy as np
    calibrator = Calibrator()
    
    # Create synthetic data
    scores = np.array([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98])
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    
    calibrator.fit(
        {"test_field": scores},
        {"test_field": labels}
    )
    
    calibrated = calibrator.calibrate({"test_field": 0.8})
    print(f"  ✅ Calibrator: Fit and calibrate works")
    print(f"     Raw 0.8 → calibrated {calibrated['test_field']:.2f}")
    
except Exception as e:
    print(f"  ❌ Validator test failed: {e}")

# ============================================================
# Test 5: Database works
# ============================================================
print("\n[5/6] Testing database...")
try:
    from app.database import db, ExtractionDB
    
    # Test insert
    extraction_id = db.insert_extraction(
        document_type="pan",
        image_path="test.jpg",
        extracted_data={"name": "Test", "pan_number": "ABCDE1234F"},
        confidence_scores={"name": 0.95, "pan_number": 0.98},
        validation_passed=True,
        status="completed",
        model_version="test"
    )
    
    assert extraction_id is not None, "Insert should return ID"
    print(f"  ✅ Database: Insert works (ID: {extraction_id})")
    
    # Test retrieve
    retrieved = db.get_extraction(extraction_id)
    assert retrieved is not None, "Should retrieve inserted record"
    assert retrieved["document_type"] == "pan"
    assert retrieved["extracted_data"]["name"] == "Test"
    print(f"  ✅ Database: Retrieve works")
    print(f"     Retrieved: {retrieved['document_type']} - {retrieved['status']}")
    
    # Test update status
    db.update_status(extraction_id, "approved", "Looks good")
    retrieved = db.get_extraction(extraction_id)
    assert retrieved["status"] == "approved"
    print(f"  ✅ Database: Status update works")
    
except Exception as e:
    print(f"  ❌ Database test failed: {e}")

# ============================================================
# Test 6: Extractor (optional — requires API key)
# ============================================================
print("\n[6/6] Testing extractor (skipped if no API key)...")
try:
    import os
    if os.getenv("OPENAI_API_KEY"):
        from app.extractor import ConstrainedExtractor
        
        extractor = ConstrainedExtractor(backend="openai")
        
        # Test with sample OCR text
        sample_text = """
        INCOME TAX DEPARTMENT
        PAN: ABCDE1234F
        Name: ARJUN MEHTA
        Father Name: RAJESH MEHTA
        DOB: 15/08/1990
        """
        
        try:
            result = extractor.extract(sample_text, "pan", model_version="test")
            print(f"  ✅ Extractor: OpenAI extraction works")
            print(f"     Extracted: {result}")
        except Exception as e:
            print(f"  ⚠️  Extractor API call failed (check key/quota): {e}")
    else:
        print("  ⏭️  Skipping extractor test (set OPENAI_API_KEY to test)")
        
except Exception as e:
    print(f"  ❌ Extractor import failed: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("All critical components tested.")
print("If you see ✅ on tests 1-5, your pipeline is ready to run.")
print("\nNext steps:")
print("  1. Set OPENAI_API_KEY if you haven't")
print("  2. Run: uvicorn app.main:app --reload")
print("  3. Open: http://localhost:8000/docs")
print("  4. Upload a real document image to /extract")