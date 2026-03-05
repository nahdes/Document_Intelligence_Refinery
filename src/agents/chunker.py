"""
src/agents/chunker.py
ChunkingEngine — Stage 3. Converts ExtractedDocument → List[LDU].
All 5 chunking rules enforced per-LDU via CES gate_chunk().
PII redacted on all text-type LDUs via CES.

Rules enforced:
  1. max_tokens        — hard token limit; text split on sentence boundaries
  2. never_split_tables — tables are atomic
  3. keep_fig_captions_with_parent — captions carry parent_chunk_id
  4. keep_lists_intact  — list items not orphaned
  5. propagate_section_headers — parent_section on all child chunks
  + resolve_cross_references — detects "see Table 3" patterns
"""
from __future__ import annotations
import hashlib
import logging
import re
from typing import Optional

from src.core.constraint_enforcement import ConstraintEnforcementSystem
from src.models.schemas import (
    ChunkType, ExtractedDocument, ExtractedTable, LDU,
)

logger = logging.getLogger(__name__)

CROSS_REF_RE = [
    re.compile(r"see\s+(Table|Figure|Section|Appendix)\s+[\dA-Z]+", re.IGNORECASE),
    re.compile(r"\((Table|Figure|Chart|Annex)\s+[\dA-Z\.]+\)", re.IGNORECASE),
]
LIST_ITEM_RE = [
    re.compile(r"^\s*[-•·▪▸*]\s+", re.MULTILINE),
    re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE),
]


class ChunkingEngine:
    """Converts ExtractedDocument → List[LDU] with all ChunkRules enforced via CES."""

    def __init__(self, ces: Optional[ConstraintEnforcementSystem] = None) -> None:
        self.ces   = ces or ConstraintEnforcementSystem()
        self.rules = self.ces.policy_engine.policy.chunk_rules

    def chunk(self, doc: ExtractedDocument) -> list[LDU]:
        """Full chunking pipeline: tables → text → figures → PII redaction."""
        ldus: list[LDU] = []

        # Step 1: Tables (atomic — Rule 2)
        for table in doc.tables:
            tbl_ldu = self._make_table_ldu(doc.doc_id, table)
            if self._validate(tbl_ldu):
                ldus.append(tbl_ldu)
                if table.caption and self.rules.keep_figure_captions_with_parent:
                    cap = self._make_caption_ldu(doc.doc_id, table.caption, table.page, tbl_ldu.chunk_id)
                    if self._validate(cap):
                        ldus.append(cap)

        # Step 2: Text blocks
        current_section: Optional[str] = None
        for block in doc.text_blocks:
            if block.is_header or (block.font_size >= 14.0 and len(block.text) < 120):
                current_section = block.text.strip()
                hdr = LDU(
                    doc_id=doc.doc_id, chunk_type=ChunkType.HEADER,
                    content=block.text.strip(),
                    token_count=self._tokens(block.text),
                    page_refs=[block.page], bbox=block.bbox,
                    parent_section=current_section,
                )
                if self._validate(hdr):
                    ldus.append(hdr)
                continue

            if self.rules.keep_lists_intact and self._is_list(block.text):
                lst = LDU(
                    doc_id=doc.doc_id, chunk_type=ChunkType.LIST,
                    content=block.text, token_count=self._tokens(block.text),
                    page_refs=[block.page], bbox=block.bbox,
                    parent_section=current_section if self.rules.propagate_section_headers else None,
                    cross_references=self._cross_refs(block.text),
                )
                if self._validate(lst):
                    ldus.append(lst)
                    continue
                # Oversized list — split into items
                for item in self._split_list(doc.doc_id, block, current_section):
                    if self._validate(item):
                        ldus.append(item)
                continue

            for part in self._split_text(block.text, self.rules.max_tokens):
                txt = LDU(
                    doc_id=doc.doc_id, chunk_type=ChunkType.TEXT,
                    content=part, token_count=self._tokens(part),
                    page_refs=[block.page], bbox=block.bbox,
                    parent_section=current_section if self.rules.propagate_section_headers else None,
                    cross_references=self._cross_refs(part) if self.rules.resolve_cross_references else [],
                )
                if self._validate(txt):
                    ldus.append(txt)

        # Step 3: Figures
        for fig in doc.figures:
            if fig.caption:
                fldu = LDU(
                    doc_id=doc.doc_id, chunk_type=ChunkType.FIGURE,
                    content=f"[Figure] {fig.caption}",
                    token_count=self._tokens(fig.caption),
                    page_refs=[fig.page], bbox=fig.bbox,
                    parent_section=current_section if self.rules.propagate_section_headers else None,
                )
                if self._validate(fldu):
                    ldus.append(fldu)

        # Step 4: PII redaction on text-type LDUs
        ldus = self._redact_all(ldus, doc.doc_id)

        logger.info(
            "ChunkingEngine: doc=%s → %d LDUs (%d text, %d table, %d header, %d list, %d figure)",
            doc.doc_id, len(ldus),
            sum(1 for l in ldus if l.chunk_type == ChunkType.TEXT),
            sum(1 for l in ldus if l.chunk_type == ChunkType.TABLE),
            sum(1 for l in ldus if l.chunk_type == ChunkType.HEADER),
            sum(1 for l in ldus if l.chunk_type == ChunkType.LIST),
            sum(1 for l in ldus if l.chunk_type == ChunkType.FIGURE),
        )
        return ldus

    def _validate(self, ldu: LDU) -> bool:
        result = self.ces.gate_chunk(ldu.doc_id, ldu.model_dump())
        if not result.passed:
            logger.debug("ChunkValidator DROP %s [%s]: %s", ldu.chunk_id[:8], result.violation_rule, result.violation_detail)
        return result.passed

    def _make_table_ldu(self, doc_id: str, table: ExtractedTable) -> LDU:
        text = self._table_to_md(table)
        return LDU(
            doc_id=doc_id, chunk_type=ChunkType.TABLE,
            content=text, token_count=self._tokens(text),
            page_refs=[table.page], bbox=table.bbox,
            table_data={"headers": table.headers, "rows": table.rows},
        )

    def _make_caption_ldu(self, doc_id: str, caption: str, page: int, parent_id: str) -> LDU:
        return LDU(
            doc_id=doc_id, chunk_type=ChunkType.CAPTION,
            content=caption, token_count=self._tokens(caption),
            page_refs=[page], parent_chunk_id=parent_id,
        )

    def _tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _split_text(self, text: str, max_tokens: int) -> list[str]:
        if self._tokens(text) <= max_tokens:
            return [text]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], []
        for sent in sentences:
            words = sent.split()
            if len(current) + len(words) > max_tokens:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                if len(words) > max_tokens:
                    for i in range(0, len(words), max_tokens):
                        chunks.append(" ".join(words[i:i + max_tokens]))
                else:
                    current = words
            else:
                current.extend(words)
        if current:
            chunks.append(" ".join(current))
        return [c for c in chunks if c.strip()] or [text]

    def _table_to_md(self, table: ExtractedTable) -> str:
        lines = []
        if table.caption:
            lines.append(f"**{table.caption}**")
        if table.headers:
            lines.append("| " + " | ".join(str(h) for h in table.headers) + " |")
            lines.append("| " + " | ".join("---" for _ in table.headers) + " |")
        for row in table.rows:
            cells = [str(c) for c in row]
            if table.headers:
                while len(cells) < len(table.headers):
                    cells.append("")
                cells = cells[:len(table.headers)]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _is_list(self, text: str) -> bool:
        return any(p.search(text) for p in LIST_ITEM_RE)

    def _cross_refs(self, text: str) -> list[str]:
        refs = []
        for p in CROSS_REF_RE:
            refs.extend(p.findall(text))
        return list(set(refs))

    def _split_list(self, doc_id: str, block, current_section: Optional[str]) -> list[LDU]:
        items = re.split(r"\n(?=\s*[-•·▪▸*\d])", block.text)
        return [
            LDU(
                doc_id=doc_id, chunk_type=ChunkType.LIST,
                content=item.strip(), token_count=self._tokens(item),
                page_refs=[block.page], bbox=block.bbox,
                parent_section=current_section if self.rules.propagate_section_headers else None,
            )
            for item in items if item.strip()
        ]

    def _redact_all(self, ldus: list[LDU], doc_id: str) -> list[LDU]:
        text_types = {ChunkType.TEXT, ChunkType.HEADER, ChunkType.CAPTION, ChunkType.LIST}
        count = 0
        for ldu in ldus:
            if ldu.chunk_type in text_types:
                redacted, records = self.ces.redact_pii(ldu.content, doc_id)
                if records:
                    ldu.content = redacted
                    ldu.content_hash = hashlib.sha256(redacted.encode()).hexdigest()
                    count += len(records)
        if count:
            logger.info("ChunkingEngine: %d PII entities redacted in doc=%s", count, doc_id)
        return ldus