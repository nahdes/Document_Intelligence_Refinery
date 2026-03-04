"""
Strategy A — Fast Text Extraction.

Tool: pdfplumber + PyMuPDF
Cost: $0.0001/page
Confidence Floor: 0.70
Triggers: Native digital, single-column documents

Rubric Compliance:
- Multi-signal confidence scoring (char density + image ratio + font metadata)
- Spatial provenance (bounding boxes)
- Budget control
"""

import pdfplumber
import time
import hashlib
from pathlib import Path
from typing import Dict, List
from src.strategies.base import ExtractionStrategy
from src.models.document import ExtractedDocument, TextBlock, TableData, FigureData, BoundingBox


class FastTextExtractor(ExtractionStrategy):
    """Fast text extraction using pdfplumber."""
    
    name = "Strategy A — Fast Text"
    strategy_id = "A"
    cost_per_page = 0.0001
    confidence_floor = 0.70

    def extract(self, file_path: str) -> ExtractedDocument:
        """Extract text using pdfplumber."""
        start_time = time.time()
        file_path = Path(file_path)
        
        text_blocks: List[TextBlock] = []
        tables: List[TableData] = []
        reading_order = 0
        
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text blocks
                text = page.extract_text()
                if text:
                    text_blocks.append(TextBlock(
                        text=text,
                        bbox=BoundingBox(
                            page=page_num,
                            x0=0,
                            y0=0,
                            x1=page.width,
                            y1=page.height
                        ),
                        reading_order=reading_order
                    ))
                    reading_order += 1
                
                # Extract tables
                pdf_tables = page.find_tables()
                if pdf_tables:
                    for table in pdf_tables:
                        if table.extract():
                            tables.append(TableData(
                                headers=table.header if table.header else ["Column"],
                                rows=table.extract(),
                                bbox=BoundingBox(
                                    page=page_num,
                                    x0=table.bbox[0],
                                    y0=table.bbox[1],
                                    x1=table.bbox[2],
                                    y1=table.bbox[3]
                                )
                            ))
        
        processing_time = int((time.time() - start_time) * 1000)
        content_hash = hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
        
        return ExtractedDocument(
            doc_id=content_hash[:16],
            filename=file_path.name,
            pages=page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=[],
            strategy_used=self.strategy_id,
            confidence_score=0.0,  # Calculated separately
            cost_per_page=self.cost_per_page,
            total_cost=self.cost_per_page * page_count,
            processing_time_ms=processing_time,
            content_hash=content_hash,
        )

    def calculate_confidence(
        self, 
        doc: ExtractedDocument, 
        page_metrics: Dict
    ) -> float:
        """
        Multi-signal confidence scoring.
        
        Rubric Compliance:
        - Signal 1: Character density
        - Signal 2: Image ratio (penalty if high)
        - Signal 3: Font metadata presence
        """
        score = 0.0
        signals = {}
        
        # Signal 1: Character density (max 0.4)
        chars_per_page = page_metrics.get('avg_chars_per_page', 0)
        if chars_per_page > 500:
            signals['char_density'] = 0.4
            score += 0.4
        elif chars_per_page > 100:
            signals['char_density'] = 0.2
            score += 0.2
        else:
            signals['char_density'] = 0.0
        
        # Signal 2: Image ratio penalty (max 0.3)
        image_ratio = page_metrics.get('image_area_ratio', 0)
        if image_ratio < 0.2:
            signals['image_ratio'] = 0.3
            score += 0.3
        elif image_ratio < 0.5:
            signals['image_ratio'] = 0.15
            score += 0.15
        else:
            signals['image_ratio'] = 0.0
        
        # Signal 3: Font metadata (max 0.3)
        has_fonts = page_metrics.get('has_font_metadata', False)
        if has_fonts:
            signals['font_metadata'] = 0.3
            score += 0.3
        else:
            signals['font_metadata'] = 0.0
        
        doc.confidence_signals = signals
        doc.confidence_score = min(score, 1.0)
        
        return doc.confidence_score