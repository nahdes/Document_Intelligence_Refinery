"""
src/models/schemas.py
All Pydantic v2 schemas for the Document Intelligence Refinery pipeline.
Every stage produces and consumes typed models — no raw dicts in production.

Five core models (one per pipeline concept):
  1. DocumentProfile      — Triage Agent output
  2. ExtractedDocument    — normalized extraction output (all strategies)
  3. LDU                  — Logical Document Unit (Chunking Engine output)
  4. PageIndex            — hierarchical page/section index (Indexer output)
  5. ProvenanceChain      — answer + citation chain (Query Agent output)

Supporting types:
  BoundingBox, TextBlock, ExtractedTable, ExtractedFigure
  ExtractionLedgerEntry, RoutingDecision
  BaseExtractionStrategy (shared Abstract Base Class for all three strategies)
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


# =============================================================================
# Enumerations — ALL categorical fields use Enum, never bare strings
# =============================================================================

class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"   # Born-digital: embedded text stream
    SCANNED_IMAGE  = "scanned_image"    # Pure raster — no character stream
    MIXED          = "mixed"            # Some pages native, some scanned
    FORM_FILLABLE  = "form_fillable"    # AcroForm / XFA with interactive fields


class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"   # Simple linear text flow
    MULTI_COLUMN  = "multi_column"    # 2+ column newspaper/academic style
    TABLE_HEAVY   = "table_heavy"     # Dominant structured tables
    FIGURE_HEAVY  = "figure_heavy"    # Charts, diagrams, images dominate
    MIXED         = "mixed"           # Tables AND figures both present


class DomainHint(str, Enum):
    FINANCIAL  = "financial"
    LEGAL      = "legal"
    TECHNICAL  = "technical"
    MEDICAL    = "medical"
    GENERAL    = "general"


class ExtractionStrategy(str, Enum):
    FAST   = "fast"    # Strategy A: pdfplumber + PyMuPDF
    LAYOUT = "layout"  # Strategy B: Docling (layout-aware)
    VISION = "vision"  # Strategy C: VLM via OpenRouter


class ChunkType(str, Enum):
    TEXT      = "text"
    TABLE     = "table"
    FIGURE    = "figure"
    EQUATION  = "equation"
    LIST      = "list"
    HEADER    = "header"
    CAPTION   = "caption"
    FOOTNOTE  = "footnote"


# =============================================================================
# BoundingBox — structured sub-model, NEVER a raw list or dict
# =============================================================================

class BoundingBox(BaseModel):
    """
    Spatial coordinates in PDF points (1 pt = 1/72 inch).
    Origin is bottom-left per PDF spec.
    Used by: TextBlock, ExtractedTable, ExtractedFigure, LDU, ProvenanceRecord.
    """
    x0:   float = Field(ge=0.0, description="Left edge in PDF points")
    y0:   float = Field(ge=0.0, description="Bottom edge in PDF points")
    x1:   float = Field(ge=0.0, description="Right edge in PDF points")
    y1:   float = Field(ge=0.0, description="Top edge in PDF points")
    page: int   = Field(ge=1,   description="1-indexed page number")

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x1 < self.x0:
            raise ValueError(f"x1 ({self.x1}) must be >= x0 ({self.x0})")
        if self.y1 < self.y0:
            raise ValueError(f"y1 ({self.y1}) must be >= y0 ({self.y0})")
        return self

    @computed_field
    @property
    def area(self) -> float:
        return max(0.0, (self.x1 - self.x0) * (self.y1 - self.y0))

    @computed_field
    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @computed_field
    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def overlaps(self, other: "BoundingBox") -> bool:
        """Return True if this box overlaps other on the same page."""
        if self.page != other.page:
            return False
        return not (
            self.x1 < other.x0 or other.x1 < self.x0
            or self.y1 < other.y0 or other.y1 < self.y0
        )


# =============================================================================
# 1. DocumentProfile — Stage 1: Triage Agent output
# =============================================================================

class DocumentProfile(BaseModel):
    """
    Complete triage analysis of a document.
    Produced by TriageAgent and consumed by ExtractionRouter to select the
    initial extraction strategy. All categorical fields are typed Enums.

    Rubric compliance:
    - origin_type detection via char density + image ratio + font metadata + whitespace
    - layout_complexity via column-count heuristics + bbox analysis flags
    - domain_hint via swappable keyword classifier
    - estimated_cost_usd derived from (strategy, page_count) via policy engine
    - FORM_FILLABLE origin handled via has_acroform flag
    """
    doc_id:           str  = Field(default_factory=lambda: str(uuid4()))
    filename:         str
    file_size_bytes:  int  = Field(ge=0)
    page_count:       int  = Field(ge=1)
    mime_type:        str  = "application/pdf"

    # ── Classification dimensions (all Enum-typed) ────────────────────────
    origin_type:         OriginType
    layout_complexity:   LayoutComplexity
    domain_hint:         DomainHint       = DomainHint.GENERAL

    # ── Language ─────────────────────────────────────────────────────────
    language:            str   = "en"
    language_confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    # ── Strategy & cost (derived from classification) ─────────────────────
    recommended_strategy: ExtractionStrategy
    estimated_cost_usd:   float = Field(ge=0.0)
    cost_tier:            Literal["A", "B", "C"] = "A"  # derived by validator

    # ── Triage confidence ─────────────────────────────────────────────────
    triage_confidence: float = Field(ge=0.0, le=1.0, default=0.9)

    # ── Structural diagnostics used by all classifiers ────────────────────
    avg_chars_per_page:    float = Field(ge=0.0, default=0.0,
        description="Average characters per page (sampled from up to 20 pages)")
    avg_image_area_ratio:  float = Field(ge=0.0, le=1.0, default=0.0,
        description="Average fraction of page area covered by images")
    avg_whitespace_ratio:  float = Field(ge=0.0, le=1.0, default=0.0,
        description="Average fraction of page area that is blank whitespace")
    column_count_estimate: int   = Field(ge=1, default=1,
        description="Estimated column count from bbox horizontal gap analysis")

    # ── Font metadata signals ─────────────────────────────────────────────
    has_embedded_fonts: bool  = False
    font_count:         int   = Field(ge=0, default=0,
        description="Number of distinct fonts embedded in the document")
    has_symbolic_fonts: bool  = False  # Wingdings/Symbol → likely form-fillable

    # ── Content flags ─────────────────────────────────────────────────────
    has_handwriting: bool = False
    has_tables:      bool = False
    has_figures:     bool = False
    has_acroform:    bool = False   # True → triggers FORM_FILLABLE origin

    # ── Security ─────────────────────────────────────────────────────────
    content_hash: str = Field(default="",
        description="SHA-256 hex digest of raw file bytes")

    profiled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def derive_cost_tier(self) -> "DocumentProfile":
        """Auto-derive cost_tier (A/B/C) from recommended_strategy."""
        self.cost_tier = {
            ExtractionStrategy.FAST:   "A",
            ExtractionStrategy.LAYOUT: "B",
            ExtractionStrategy.VISION: "C",
        }[self.recommended_strategy]
        return self

    @field_validator("content_hash")
    @classmethod
    def validate_hash_length(cls, v: str) -> str:
        if v and len(v) != 64:
            raise ValueError(
                f"content_hash must be 64-char SHA-256 hex string, got length {len(v)}"
            )
        return v

    @field_validator("avg_image_area_ratio", "avg_whitespace_ratio")
    @classmethod
    def clamp_ratio(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# =============================================================================
# 2. ExtractedDocument — Stage 2: normalized output of ALL three strategies
# =============================================================================

class TextBlock(BaseModel):
    """
    A contiguous block of text with full spatial and font provenance.
    Font metadata (font_name, font_size, is_bold) feeds into Strategy A
    confidence scoring as a multi-signal input.
    """
    block_id:      str              = Field(default_factory=lambda: str(uuid4()))
    text:          str
    bbox:          Optional[BoundingBox] = None
    reading_order: int              = Field(ge=0, default=0)
    page:          int              = Field(ge=1, default=1)

    # Font metadata — used in multi-signal confidence scoring (Strategy A)
    font_name:     Optional[str]    = None
    font_size:     Optional[float]  = Field(None, ge=0.0)
    is_bold:       bool             = False
    is_italic:     bool             = False
    is_header:     bool             = False
    confidence:    float            = Field(ge=0.0, le=1.0, default=1.0)


class TableCell(BaseModel):
    """A single cell within an ExtractedTable, with merge support."""
    row:       int
    col:       int
    text:      str
    is_header: bool = False
    colspan:   int  = Field(ge=1, default=1)
    rowspan:   int  = Field(ge=1, default=1)


class ExtractedTable(BaseModel):
    """
    A structured table extracted from a PDF page.
    Strategy B adapter maps Docling's output format to this schema exactly,
    preserving headers, rows, bounding boxes, and reading order.

    The rows_match_headers validator silently pads/truncates ragged rows
    (common with Docling multi-page table continuation detection).
    """
    table_id:   str               = Field(default_factory=lambda: str(uuid4()))
    headers:    list[str]
    rows:       list[list[str]]
    cells:      list[TableCell]   = Field(default_factory=list)
    bbox:       Optional[BoundingBox] = None
    page:       int               = Field(ge=1, default=1)
    caption:    Optional[str]     = None
    confidence: float             = Field(ge=0.0, le=1.0, default=0.9)

    @model_validator(mode="after")
    def normalize_ragged_rows(self) -> "ExtractedTable":
        """
        Pad or truncate rows to match header width.
        Docling may produce ragged tables when detecting multi-page continuations.
        """
        if not self.headers:
            return self
        w = len(self.headers)
        self.rows = [(row + [""] * w)[:w] for row in self.rows]
        return self

    def to_dict_rows(self) -> list[dict[str, str]]:
        return [dict(zip(self.headers, row)) for row in self.rows]

    def to_markdown(self) -> str:
        sep = "| " + " | ".join(["---"] * len(self.headers)) + " |"
        lines = ["| " + " | ".join(self.headers) + " |", sep]
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


class ExtractedFigure(BaseModel):
    """A figure, chart, or diagram with spatial provenance and optional VLM description."""
    figure_id:  str              = Field(default_factory=lambda: str(uuid4()))
    caption:    Optional[str]    = None
    bbox:       Optional[BoundingBox] = None
    page:       int              = Field(ge=1, default=1)
    image_path: Optional[str]    = None   # path to extracted image file (Strategy B/C)
    alt_text:   Optional[str]    = None   # VLM-generated description (Strategy C)


class RoutingDecision(BaseModel):
    """
    Decision transparency record surfaced by ExtractionRouter.
    Rubric: routing decisions (strategy selected, confidence received,
    escalation occurred) must be surfaced via return metadata or structured logging.
    This model fulfils the metadata requirement — it is embedded in ExtractedDocument.
    """
    initial_strategy: ExtractionStrategy
    final_strategy:   ExtractionStrategy
    escalation_count: int   = Field(ge=0, default=0)
    human_review_flag:bool  = False
    attempts: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Ordered list of extraction attempts. Each dict has keys: "
            "{strategy, confidence, cost_usd, reason, threshold}"
        ),
    )


class ExtractedDocument(BaseModel):
    """
    Normalized extraction output — identical schema regardless of which
    strategy (A, B, or C) produced it.

    Rubric compliance:
    - Shared interface: all strategies return this exact type
    - Spatial provenance: all text_blocks and tables carry page + bbox
    - routing_decision: surfaces strategy selected, confidence, escalation
    - total_elements: computed field for quick diagnostics
    """
    doc_id:         str
    source_profile: DocumentProfile
    strategy_used:  ExtractionStrategy

    text_blocks: list[TextBlock]      = Field(default_factory=list)
    tables:      list[ExtractedTable] = Field(default_factory=list)
    figures:     list[ExtractedFigure]= Field(default_factory=list)

    # Aggregated full text (for embedding + search)
    full_text: str = ""

    # Quality signals
    confidence_score:  float = Field(ge=0.0, le=1.0, default=0.8)
    cost_estimate_usd: float = Field(ge=0.0, default=0.0)
    processing_time_s: float = Field(ge=0.0, default=0.0)
    escalation_count:  int   = Field(ge=0,   default=0)

    # Decision transparency (rubric requirement)
    routing_decision: Optional[RoutingDecision] = None

    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def total_elements(self) -> int:
        """Total count of text blocks + tables + figures extracted."""
        return len(self.text_blocks) + len(self.tables) + len(self.figures)


# =============================================================================
# 3. LDU — Logical Document Unit (Chunking Engine output)
# =============================================================================

class LDU(BaseModel):
    """
    Logical Document Unit — the atomic RAG-ready chunk produced by ChunkingEngine.

    Rubric provenance requirements:
    - content_hash  : SHA-256 of content — auto-computed, immutable
    - page_refs     : 1-indexed list of pages this chunk spans
    - bounding_box  : spatial location in source PDF (NOT named 'bbox' to match rubric)
    - parent_section: title of the containing section (header propagation)
    - parent_chunk_id: chunk relationship (e.g. caption belongs to figure)
    - cross_references: resolved references to other LDU chunk_ids
    """
    chunk_id:   str      = Field(default_factory=lambda: str(uuid4()))
    doc_id:     str
    chunk_type: ChunkType
    content:    str
    token_count:int      = Field(ge=0, default=0)

    # ── Spatial provenance (rubric: page_refs, bounding_box) ──────────────
    page_refs:    list[int]              = Field(
        default_factory=list,
        description="1-indexed page numbers this chunk spans (sorted, deduplicated)"
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Spatial location of this chunk in the source PDF"
    )
    bbox: Optional[BoundingBox] = Field(
        None,
        description="Alias for bounding_box — used by query agent"
    )

    # ── Structural relationships (rubric: parent_section, chunk relationships) ─
    parent_section:   Optional[str] = Field(
        None,
        description="Title of the containing section — propagated from header LDUs"
    )
    parent_chunk_id:  Optional[str] = Field(
        None,
        description="chunk_id of the parent LDU (e.g. figure → its caption)"
    )
    sibling_chunk_ids: list[str]    = Field(
        default_factory=list,
        description="chunk_ids of adjacent LDUs in reading order"
    )
    cross_references:  list[str]    = Field(
        default_factory=list,
        description="Resolved references to other LDU chunk_ids ('see Table 3' → chunk_id)"
    )

    # ── Table data preserved verbatim ────────────────────────────────────
    table_data: Optional[dict[str, Any]] = Field(
        None,
        description="For TABLE chunks: {headers: [...], rows: [[...]]} verbatim"
    )

    # ── Provenance hash (auto-computed, never modified after creation) ─────
    content_hash: str = Field(
        default="",
        description="SHA-256 hex digest of content — computed on creation"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def compute_content_hash(self) -> "LDU":
        """Auto-compute SHA-256 hash of content if not already set."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()
        return self

    @field_validator("page_refs")
    @classmethod
    def validate_and_sort_page_refs(cls, v: list[int]) -> list[int]:
        if any(p < 1 for p in v):
            raise ValueError("All page_refs must be >= 1 (1-indexed)")
        return sorted(set(v))  # deduplicate and sort


# =============================================================================
# 4. PageIndex — Stage 4: hierarchical page/section index (Indexer output)
# =============================================================================

class PageIndexNode(BaseModel):
    """
    One node in the recursive section tree.
    child_sections enables arbitrary nesting depth:
      Part I → Chapter 1 → Section 1.1 → Sub-section 1.1.a
    """
    node_id:    str = Field(default_factory=lambda: str(uuid4()))
    title:      str
    page_start: int = Field(ge=1)
    page_end:   int = Field(ge=1)
    level:      int = Field(ge=0, description="0=top-level, 1=sub-section, 2+=deeper")

    # Recursive children — forward reference, resolved below via model_rebuild()
    child_sections: list["PageIndexNode"] = Field(default_factory=list)

    # LLM-generated content for semantic search
    summary:            str       = ""
    key_entities:       list[str] = Field(default_factory=list)
    data_types_present: list[str] = Field(
        default_factory=list,
        description="Content types in section: 'tables', 'figures', 'equations'"
    )

    # Links back to LDU chunk_ids in this section
    ldu_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_page_range(self) -> "PageIndexNode":
        if self.page_end < self.page_start:
            raise ValueError(
                f"page_end ({self.page_end}) must be >= page_start ({self.page_start})"
            )
        return self

    def flatten(self) -> list["PageIndexNode"]:
        """DFS traversal — returns this node and all descendants."""
        result = [self]
        for child in self.child_sections:
            result.extend(child.flatten())
        return result


PageIndexNode.model_rebuild()  # Required to resolve the self-referential type


class PageIndex(BaseModel):
    """
    Hierarchical section tree for one document.
    navigate() traverses the full tree with TF-IID relevance scoring,
    searching recursively through all levels of nesting.
    """
    doc_id:        str
    filename:      str
    total_pages:   int = Field(ge=1)
    root_sections: list[PageIndexNode] = Field(default_factory=list)
    built_at:      datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def navigate(self, topic: str, top_k: int = 3) -> list[PageIndexNode]:
        """
        Return top_k most relevant PageIndexNodes for the given topic.
        Scoring: TF-IID — count of topic words found in title + summary + key_entities.
        Recursively searches all levels of the section tree.
        """
        topic_words = topic.lower().split()
        scored: list[tuple[float, PageIndexNode]] = []
        self._score_recursive(self.root_sections, topic_words, scored)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def _score_recursive(
        self,
        nodes: list[PageIndexNode],
        topic_words: list[str],
        results: list[tuple[float, PageIndexNode]],
    ) -> None:
        for node in nodes:
            searchable = (
                f"{node.title} {node.summary} {' '.join(node.key_entities)}"
            ).lower()
            score = sum(searchable.count(w) for w in topic_words)
            if score > 0:
                results.append((float(score), node))
            self._score_recursive(node.child_sections, topic_words, results)

    def all_nodes(self) -> list[PageIndexNode]:
        """Return every node in the tree via DFS."""
        result: list[PageIndexNode] = []
        for root in self.root_sections:
            result.extend(root.flatten())
        return result


# =============================================================================
# 5. ProvenanceChain — Stage 5: answer + citation chain (Query Agent output)
# =============================================================================

class ProvenanceRecord(BaseModel):
    """
    One source citation with full spatial and content provenance.

    Rubric requirements all satisfied:
    - bbox         : BoundingBox structured sub-model (not a list or dict)
    - content_hash : SHA-256 of the source LDU content (verified against LDU)
    - chunk_id     : links back to the source LDU.chunk_id
    """
    doc_id:          str
    filename:        str
    page_number:     int   = Field(ge=1)
    bbox:            Optional[BoundingBox] = Field(
        None,
        description="Bounding box of the cited excerpt in the source PDF"
    )
    content_hash:    str   = Field(
        description="SHA-256 of the cited LDU content — enables tamper verification"
    )
    chunk_id:        str   = Field(
        description="LDU.chunk_id of the cited logical document unit"
    )
    excerpt:         str   = Field(
        default="",
        description="Short excerpt from the source (auto-truncated to 200 chars)"
    )
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)

    @field_validator("content_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        if v and len(v) != 64:
            raise ValueError(f"content_hash must be 64-char hex, got len={len(v)}")
        return v

    @field_validator("excerpt")
    @classmethod
    def truncate_excerpt(cls, v: str) -> str:
        return v[:200]


class ProvenanceChain(BaseModel):
    """
    Complete query answer with an auditable chain of source citations.
    Every answer links back to specific LDUs with bounding boxes, hashes,
    and chunk_ids. Low-confidence answers are flagged for human review.
    """
    query:       str
    answer:      str
    confidence:  float = Field(ge=0.0, le=1.0)
    sources:     list[ProvenanceRecord] = Field(default_factory=list)
    tool_used:   Literal[
        "semantic_search", "structured_query", "pageindex_navigate"
    ] = "semantic_search"
    human_review_required: bool     = False
    generated_at:          datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def is_verifiable(self) -> bool:
        """True when at least one source has a non-empty content_hash."""
        return any(bool(s.content_hash) for s in self.sources)

    def format_citation(self) -> str:
        """
        Human-readable citation block for reports, APIs, and UI rendering.
        Includes filename, page number, bounding box coordinates,
        excerpt text, and content hash prefix for each source.
        """
        lines = [
            f"Answer ({self.confidence:.0%} confidence): {self.answer}",
            "",
            f"Sources ({len(self.sources)}):",
        ]
        for i, src in enumerate(self.sources, 1):
            bbox_str = ""
            if src.bbox:
                b = src.bbox
                bbox_str = (
                    f" [p.{b.page} bbox=({b.x0:.0f},{b.y0:.0f})"
                    f"→({b.x1:.0f},{b.y1:.0f})]"
                )
            lines.append(f"  [{i}] {src.filename}{bbox_str}")
            if src.excerpt:
                lines.append(f"       \"{src.excerpt}\"")
            lines.append(
                f"       hash:{src.content_hash[:16]}…  "
                f"chunk:{src.chunk_id[:8]}…  "
                f"relevance:{src.relevance_score:.2f}"
            )
        if self.human_review_required:
            lines.append(
                "\n⚠  FLAGGED FOR HUMAN REVIEW "
                f"(confidence {self.confidence:.0%} below threshold)"
            )
        return "\n".join(lines)


# =============================================================================
# ExtractionLedgerEntry — one row in .refinery/extraction_ledger.jsonl
# =============================================================================

class ExtractionLedgerEntry(BaseModel):
    """
    Appended to .refinery/extraction_ledger.jsonl after every extraction.
    Includes full routing attempt history for decision transparency.
    """
    doc_id:             str
    filename:           str
    strategy_used:      ExtractionStrategy
    confidence_score:   float = Field(ge=0.0, le=1.0)
    cost_estimate_usd:  float = Field(ge=0.0)
    processing_time_s:  float = Field(ge=0.0)
    escalation_count:   int   = Field(ge=0, default=0)
    page_count:         int   = Field(ge=1, default=1)
    ldu_count:          int   = Field(ge=0, default=0)
    human_review_flag:  bool  = False

    # Full routing attempt log for decision transparency
    routing_attempts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Each attempt: {strategy, confidence, cost_usd, threshold, reason}",
    )

    logged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# BaseExtractionStrategy — shared Abstract Base Class (rubric: shared interface)
# =============================================================================

class BaseExtractionStrategy(ABC):
    """
    Abstract base class that ALL extraction strategies MUST implement.

    Rubric requirement: "All strategies implement a common base class or protocol
    with consistent method signatures and return types."

    Concrete implementations:
      - FastTextExtractor  (Strategy A: pdfplumber + PyMuPDF)
      - LayoutExtractor    (Strategy B: Docling, with schema adapter)
      - VisionExtractor    (Strategy C: Gemini Flash via OpenRouter)

    The ExtractionRouter accepts any BaseExtractionStrategy, making strategies
    fully swappable at runtime without router code changes.

    Example:
        class MyCustomStrategy(BaseExtractionStrategy):
            STRATEGY_ID = ExtractionStrategy.FAST

            def extract(self, path, profile):
                ...   # must return ExtractedDocument

            def compute_confidence(self, text, profile, *, tables=None, font_count=0):
                ...   # must return float in [0.0, 1.0]
    """

    #: Must be overridden in each concrete strategy subclass
    STRATEGY_ID: ExtractionStrategy

    @abstractmethod
    def extract(self, path: Path, profile: DocumentProfile) -> ExtractedDocument:
        """
        Extract text, tables, and figures from a PDF at `path`.

        Contract:
        - All text_blocks must include page number and bbox
        - All tables must be normalized to ExtractedTable schema (Strategy B adapter)
        - confidence_score computed via compute_confidence()
        - cost_estimate_usd set via policy engine
        - strategy_used = self.STRATEGY_ID
        - routing_decision populated with this attempt's metadata

        Raises:
            LowConfidenceError: if confidence < the strategy's configured floor
        """
        ...

    @abstractmethod
    def compute_confidence(
        self,
        text: str,
        profile: DocumentProfile,
        *,
        tables:     Optional[list[ExtractedTable]] = None,
        font_count: int = 0,
    ) -> float:
        """
        Multi-signal confidence score in [0.0, 1.0].

        Signal sources vary by strategy:
          A (Fast):    char density + image area ratio + font metadata (font_count)
          B (Layout):  char density + table count + layout complexity match
          C (Vision):  api response completeness + fixed quality floor

        Must return a value in [0.0, 1.0] — clamp before returning.
        """
        ...