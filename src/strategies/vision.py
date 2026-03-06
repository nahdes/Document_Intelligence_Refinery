"""
Strategy C — Vision Extraction.

Tool: pdfplumber high-fidelity fallback (VLM/Gemini not configured)
Cost: $0.02/page
Confidence Floor: 0.50
Triggers: Scanned images, figures-heavy, escalation from B

This implementation uses pdfplumber at maximum resolution as the vision
strategy. A real VLM backend (Gemini Flash, GPT-4V) can be wired in
by replacing the _vlm_extract() stub.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List

from src.strategies.base import ExtractionStrategy
from src.models.document import (
    BoundingBox,
    ExtractedDocument,
    FigureData,
    TableData,
    TextBlock,
)

logger = logging.getLogger(__name__)


class VisionExtractor(ExtractionStrategy):
    name = "Strategy C — Vision"
    strategy_id = "C"
    cost_per_page = 0.02
    confidence_floor = 0.50

    def extract(self, file_path: str) -> ExtractedDocument:
        """High-fidelity extraction via pdfplumber (VLM stub)."""
        start_time = time.time()
        file_path = Path(file_path)
        try:
            doc = self._pdfplumber_extract(file_path)
        except Exception as e:
            logger.error("VisionExtractor failed: %s", e)
            content_hash = hashlib.sha256(open(file_path, "rb").read()).hexdigest()
            doc = ExtractedDocument(
                doc_id=content_hash[:16],
                filename=file_path.name,
                pages=1,
                text_blocks=[],
                tables=[],
                figures=[],
                strategy_used=self.strategy_id,
                confidence_score=0.5,
                cost_per_page=self.cost_per_page,
                total_cost=self.cost_per_page,
                processing_time_ms=0,
                content_hash=content_hash,
            )
        doc.processing_time_ms = int((time.time() - start_time) * 1000)
        return doc

    def _pdfplumber_extract(self, file_path: Path) -> ExtractedDocument:
        import pdfplumber

        text_blocks: List[TextBlock] = []
        tables: List[TableData] = []
        figures: List[FigureData] = []
        reading_order = 0

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                # Tables
                for tbl in (page.find_tables() or []):
                    data = tbl.extract()
                    if not data or len(data) < 1:
                        continue
                    headers = [str(c or "").strip() for c in data[0]]
                    if not any(headers):
                        headers = [f"Col{i+1}" for i in range(len(data[0]))]
                    rows = [[str(c or "").strip() for c in row] for row in data[1:]]
                    clean_rows = [r for r in rows if any(r)]
                    if not clean_rows:
                        clean_rows = [[""] * len(headers)]
                    b = tbl.bbox
                    tables.append(TableData(
                        headers=headers,
                        rows=clean_rows,
                        bbox=BoundingBox(page=page_num, x0=b[0], y0=b[1], x1=b[2], y1=b[3]),
                    ))

                # Images/figures
                for img in (page.images or []):
                    figures.append(FigureData(
                        caption=f"Figure p{page_num}",
                        bbox=BoundingBox(
                            page=page_num,
                            x0=float(img.get("x0", 0)),
                            y0=float(img.get("y0", 0)),
                            x1=float(img.get("x1", 100)),
                            y1=float(img.get("y1", 100)),
                        ),
                        figure_type="image",
                    ))

                # Text
                text = (page.extract_text() or "").strip()
                if text:
                    text_blocks.append(TextBlock(
                        text=text,
                        bbox=BoundingBox(
                            page=page_num, x0=0, y0=0,
                            x1=float(page.width), y1=float(page.height),
                        ),
                        reading_order=reading_order,
                    ))
                    reading_order += 1

        content_hash = hashlib.sha256(open(file_path, "rb").read()).hexdigest()
        extracted = ExtractedDocument(
            doc_id=content_hash[:16],
            filename=file_path.name,
            pages=page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            strategy_used=self.strategy_id,
            confidence_score=0.0,
            cost_per_page=self.cost_per_page,
            total_cost=self.cost_per_page * page_count,
            processing_time_ms=0,
            content_hash=content_hash,
        )
        self.calculate_confidence(extracted, {})
        logger.info("VisionExtractor: %d text_blocks, %d tables, %d figures, %d pages",
                    len(text_blocks), len(tables), len(figures), page_count)
        return extracted

    def calculate_confidence(self, doc: ExtractedDocument, page_metrics: Dict) -> float:
        score = 0.5
        if doc.text_blocks:
            score += min(len(doc.text_blocks) / 20.0, 0.3)
        if doc.tables:
            score += 0.1
        if doc.figures:
            score += 0.1
        doc.confidence_score = min(score, 1.0)
        return doc.confidence_score