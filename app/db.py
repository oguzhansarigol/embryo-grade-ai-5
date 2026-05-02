"""SQLite-backed prediction history (doc §6 — clinical follow-up)."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    image_filename    TEXT NOT NULL,
    predicted_class   TEXT NOT NULL,
    confidence        REAL NOT NULL,
    warning_flag      INTEGER NOT NULL,
    gradcam_path      TEXT,
    actual_class      TEXT,
    pregnancy_outcome TEXT
);
"""


class HistoryDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert(self, image_filename: str, predicted_class: str, confidence: float,
               warning_flag: bool, gradcam_path: Optional[str] = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (timestamp, image_filename, predicted_class, confidence,
                    warning_flag, gradcam_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (datetime.utcnow().isoformat(timespec="seconds"),
                 image_filename, predicted_class, confidence,
                 int(warning_flag), gradcam_path),
            )
            return cur.lastrowid

    def list_all(self, limit: int = 200) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()

    def update_followup(self, pred_id: int, actual_class: Optional[str],
                        pregnancy_outcome: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE predictions
                   SET actual_class = ?, pregnancy_outcome = ?
                   WHERE id = ?""",
                (actual_class, pregnancy_outcome, pred_id),
            )

    def export_csv(self) -> str:
        rows = self.list_all(limit=10_000)
        if not rows:
            return ""
        cols = rows[0].keys()
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join("" if r[c] is None else str(r[c]).replace(",", ";")
                                  for c in cols))
        return "\n".join(lines)
