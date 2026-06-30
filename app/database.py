"""
Database layer for logging extractions and storing results.
Start with SQLite (zero setup), swap to Postgres later.
"""

import sqlite3
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

DB_PATH = Path("data/extractions.db")


class ExtractionDB:
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extractions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_type TEXT NOT NULL,
                    image_path TEXT,
                    raw_text_hash TEXT,
                    extracted_data JSON,
                    confidence_scores JSON,
                    validation_passed BOOLEAN,
                    status TEXT DEFAULT 'pending',
                    errors TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    extraction_id INTEGER,
                    event TEXT,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def insert_extraction(
        self,
        document_type: str,
        image_path: str,
        extracted_data: Dict[str, Any],
        confidence_scores: Optional[Dict[str, float]] = None,
        validation_passed: bool = False,
        status: str = "pending",
        errors: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> int:
        """Store extraction result. Returns the row ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO extractions
                (document_type, image_path, extracted_data, confidence_scores,
                 validation_passed, status, errors, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_type,
                    image_path,
                    json.dumps(extracted_data, default=str),
                    json.dumps(confidence_scores) if confidence_scores else None,
                    validation_passed,
                    status,
                    errors,
                    model_version,
                ),
            )
            return cursor.lastrowid

    def get_extraction(self, extraction_id: int) -> Optional[Dict]:
        """Retrieve a stored extraction by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM extractions WHERE id = ?", (extraction_id,)
            ).fetchone()

            if row is None:
                return None

            result = dict(row)
            result["extracted_data"] = json.loads(result["extracted_data"])
            if result["confidence_scores"]:
                result["confidence_scores"] = json.loads(result["confidence_scores"])
            return result

    def get_pending_reviews(self) -> List[Dict]:
        """Get extractions that need human review."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM extractions WHERE status = 'pending_review'"
            ).fetchall()

            results = []
            for row in rows:
                result = dict(row)
                result["extracted_data"] = json.loads(result["extracted_data"])
                if result["confidence_scores"]:
                    result["confidence_scores"] = json.loads(result["confidence_scores"])
                results.append(result)

            return results

    def update_status(self, extraction_id: int, status: str, notes: Optional[str] = None):
        """Update extraction status (e.g., after human review)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE extractions SET status = ? WHERE id = ?",
                (status, extraction_id),
            )
            if notes:
                conn.execute(
                    "INSERT INTO audit_log (extraction_id, event, details) VALUES (?, ?, ?)",
                    (extraction_id, "status_update", notes),
                )


db = ExtractionDB()
