"""
Strategy B — Layout-Aware Extraction.

Tool: Docling (IBM)
Cost: $0.005/page
Confidence Floor: 0.60
Triggers: Multi-column, table-heavy, mixed origin
"""

import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List
from src.strategies.base import ExtractionStrategy
from src.models.document import ExtractedDocument, TextBlock, TableData, FigureData, BoundingBox

logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


class LayoutExtractor(ExtractionStrategy):
    name = "Strategy B — Layout-Aware"
    strategy_id = "B"
    cost_per_page = 0.005
    confidence_floor = 0.60

    def __init__(self):
        if DOCLING_AVAILABLE:
            try:
                # Disable OCR and image processing — not needed for native digital PDFs
                # and causes std::bad_alloc on large documents via RapidOCR
                opts = PdfPipelineOptions()
                opts.do_ocr = False
                opts.do_table_structure = True
                opts.generate_picture_images = False
                self.converter = DocumentConverter(
                    allowed_formats=None,
                )
                # Store options to pass at convert time
                self._pipeline_opts = opts
            except Exception as e:
                logger.warning("Docling init failed: %s", e)
                self.converter = None
                self._pipeline_opts = None
        else:
            self.converter = None
            self._pipeline_opts = None

    def _normalize_docling_output(self, result, file_path: Path) -> ExtractedDocument:
        # Unwrap ConversionResult -> DoclingDocument
        doc_obj = getattr(result, "document", result)

        text_blocks: List[TextBlock] = []
        tables: List[TableData] = []
        figures: List[FigureData] = []
        reading_order = 0

        # Text: try result.document.texts (modern API), fallback to export_to_markdown
        texts_iter = None
        for attr in ("texts", "text_elements"):
            if hasattr(doc_obj, attr):
                try:
                    texts_iter = list(getattr(doc_obj, attr))
                    break
                except Exception:
                    pass

        if texts_iter:
            for item in texts_iter:
                text = getattr(item, "text", None)
                if not text or not str(text).strip():
                    continue
                page_no, x0, y0, x1, y1 = _extract_prov(item)
                text_blocks.append(TextBlock(
                    text=str(text).strip(),
                    bbox=BoundingBox(page=page_no, x0=x0, y0=y0, x1=x1, y1=y1),
                    reading_order=reading_order,
                ))
                reading_order += 1
        else:
            # Last resort: export_to_markdown -> split paragraphs
            try:
                md = doc_obj.export_to_markdown()
                for i, para in enumerate(md.split("\n\n")):
                    para = para.strip()
                    if para:
                        text_blocks.append(TextBlock(
                            text=para,
                            bbox=BoundingBox(page=1, x0=0, y0=0, x1=100, y1=100),
                            reading_order=i,
                        ))
            except Exception as e:
                logger.warning("Docling markdown export failed: %s", e)

        # Tables
        for tbl in (getattr(doc_obj, "tables", None) or []):
            headers, rows = _extract_table_data(tbl)
            if not headers:
                continue
            page_no, x0, y0, x1, y1 = _extract_prov(tbl)
            caption_raw = getattr(tbl, "caption", None)
            clean_rows = [r for r in rows if any(c.strip() for c in r)] if rows else []
            if not clean_rows:
                clean_rows = [[""] * len(headers)]
            tables.append(TableData(
                headers=headers,
                rows=clean_rows,
                bbox=BoundingBox(page=page_no, x0=x0, y0=y0, x1=x1, y1=y1),
                caption=_safe_str(caption_raw) or None,
            ))

        # Figures
        for fig in (getattr(doc_obj, "pictures", None) or []):
            page_no, x0, y0, x1, y1 = _extract_prov(fig)
            figures.append(FigureData(
                caption=_safe_str(getattr(fig, "caption", "Figure")) or "Figure",
                bbox=BoundingBox(page=page_no, x0=x0, y0=y0, x1=x1, y1=y1),
                figure_type="image",
            ))

        # Page count
        page_count = 1
        for attr in ("pages", "num_pages"):
            val = getattr(doc_obj, attr, None)
            if val is not None:
                try:
                    page_count = len(val) if hasattr(val, "__len__") else int(val)
                    break
                except Exception:
                    pass
        if page_count == 1 and text_blocks:
            page_count = max((b.bbox.page for b in text_blocks if b.bbox), default=1)

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
        logger.info("LayoutExtractor(docling): %d text_blocks, %d tables, %d figs, %d pages",
                    len(text_blocks), len(tables), len(figures), page_count)
        return extracted

    def extract(self, file_path: str) -> ExtractedDocument:
        start_time = time.time()
        file_path = Path(file_path)
        if not DOCLING_AVAILABLE or self.converter is None:
            return self._pdfplumber_extract(file_path)
        try:
            from docling.datamodel.document import ConversionResult
            if self._pipeline_opts is not None:
                result = self.converter.convert(
                    str(file_path),
                    raises_on_error=False,
                )
            else:
                result = self.converter.convert(str(file_path), raises_on_error=False)
            doc = self._normalize_docling_output(result, file_path)
            doc.processing_time_ms = int((time.time() - start_time) * 1000)
            return doc
        except Exception as e:
            logger.warning("Docling extraction failed (%s), falling back to pdfplumber", e)
            return self._pdfplumber_extract(file_path)

    def _pdfplumber_extract(self, file_path: Path) -> ExtractedDocument:
        import pdfplumber
        text_blocks: List[TextBlock] = []
        tables: List[TableData] = []
        reading_order = 0

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                # Tables
                for tbl in (page.find_tables() or []):
                    data = tbl.extract()
                    if not data or len(data) < 2:
                        continue
                    headers = [str(c or "").strip() for c in data[0]]
                    if not any(headers):
                        headers = [f"Col{i+1}" for i in range(len(data[0]))]
                    rows = [[str(c or "").strip() for c in row] for row in data[1:]]
                    b = tbl.bbox
                    clean_rows = [r for r in rows if any(r)]
                    if not clean_rows:
                        clean_rows = [[""] * len(headers)]
                    tables.append(TableData(
                        headers=headers,
                        rows=clean_rows,
                        bbox=BoundingBox(page=page_num, x0=b[0], y0=b[1], x1=b[2], y1=b[3]),
                    ))

                # Text - extract words and group into paragraphs
                words = page.extract_words(keep_blank_chars=False, use_text_flow=True) or []
                if not words:
                    text = (page.extract_text() or "").strip()
                    if text:
                        text_blocks.append(TextBlock(
                            text=text,
                            bbox=BoundingBox(page=page_num, x0=0, y0=0,
                                             x1=float(page.width), y1=float(page.height)),
                            reading_order=reading_order,
                        ))
                        reading_order += 1
                    continue

                # Group into lines by y proximity
                lines: List[List[dict]] = []
                for w in words:
                    placed = False
                    for line in lines:
                        if abs(w["top"] - line[0]["top"]) < 5:
                            line.append(w)
                            placed = True
                            break
                    if not placed:
                        lines.append([w])

                # Group lines into paragraphs by vertical gap
                paragraphs: List[List[List[dict]]] = []
                for line in sorted(lines, key=lambda l: l[0]["top"]):
                    if not paragraphs:
                        paragraphs.append([line])
                        continue
                    last_y = paragraphs[-1][-1][0]["top"]
                    line_h = paragraphs[-1][-1][0].get("height", 12)
                    if (line[0]["top"] - last_y) > line_h * 1.8:
                        paragraphs.append([line])
                    else:
                        paragraphs[-1].append(line)

                for para in paragraphs:
                    para_text = " ".join(
                        " ".join(w["text"] for w in line) for line in para
                    ).strip()
                    if not para_text:
                        continue
                    all_w = [w for line in para for w in line]
                    text_blocks.append(TextBlock(
                        text=para_text,
                        bbox=BoundingBox(
                            page=page_num,
                            x0=min(w["x0"] for w in all_w),
                            y0=min(w["top"] for w in all_w),
                            x1=max(w["x1"] for w in all_w),
                            y1=max(w["bottom"] for w in all_w),
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
            figures=[],
            strategy_used=self.strategy_id,
            confidence_score=0.0,
            cost_per_page=self.cost_per_page,
            total_cost=self.cost_per_page * page_count,
            processing_time_ms=0,
            content_hash=content_hash,
        )
        self.calculate_confidence(extracted, {})
        logger.info("LayoutExtractor(pdfplumber): %d text_blocks, %d tables, %d pages",
                    len(text_blocks), len(tables), page_count)
        return extracted

    def calculate_confidence(self, doc: ExtractedDocument, page_metrics: Dict) -> float:
        score = 0.5
        signals = {}
        if doc.text_blocks:
            signals["text_extracted"] = min(len(doc.text_blocks) / 10.0, 0.3)
            score += signals["text_extracted"]
        else:
            signals["text_extracted"] = 0.0
        signals["tables_detected"] = 0.15 if doc.tables else 0.0
        score += signals["tables_detected"]
        signals["figures_detected"] = 0.05 if doc.figures else 0.0
        score += signals["figures_detected"]
        doc.confidence_signals = signals
        doc.confidence_score = min(score, 1.0)
        return doc.confidence_score


def _extract_prov(item):
    page_no, x0, y0, x1, y1 = 1, 0.0, 0.0, 100.0, 100.0
    prov = getattr(item, "prov", None)
    if prov:
        p = prov[0] if isinstance(prov, (list, tuple)) and prov else prov
        page_no = int(getattr(p, "page_no", 0)) + 1
        bbox = getattr(p, "bbox", None)
        if bbox is not None:
            x0 = float(getattr(bbox, "l", getattr(bbox, "x0", 0)))
            y0 = float(getattr(bbox, "t", getattr(bbox, "y0", 0)))
            x1 = float(getattr(bbox, "r", getattr(bbox, "x1", 100)))
            y1 = float(getattr(bbox, "b", getattr(bbox, "y1", 100)))
            # PDF coords have origin bottom-left; t > b is normal — ensure y0 < y1
            if y0 > y1:
                y0, y1 = y1, y0
            if x0 > x1:
                x0, x1 = x1, x0
    return page_no, x0, y0, x1, y1


def _extract_table_data(tbl):
    # Modern Docling: tbl.data.grid = List[List[TableCell]]
    grid = None
    data_attr = getattr(tbl, "data", None)
    if data_attr is not None:
        grid = getattr(data_attr, "grid", None)
    if grid is None:
        grid = getattr(tbl, "grid", None)
    if grid and len(grid) >= 1:
        def ct(cell): return str(getattr(cell, "text", "") or "").strip()
        headers = [ct(c) for c in grid[0]]
        rows = [[ct(c) for c in row] for row in grid[1:]]
        if any(headers):
            return headers, rows
    # Legacy fallback
    headers = getattr(tbl, "headers", None)
    rows = getattr(tbl, "rows", getattr(tbl, "data", []))
    if headers:
        return list(headers), [list(r) for r in rows]
    return [], []


def _safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    return str(getattr(val, "text", val)).strip()