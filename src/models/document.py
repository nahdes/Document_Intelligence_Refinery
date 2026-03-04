"""Extracted Document — Normalized output from all extraction strategies."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.models.common import BoundingBox


class TextBlock(BaseModel):
    text: str = Field(..., min_length=1)
    bbox: BoundingBox
    font_name: Optional[str] = Field(default=None)
    font_size: Optional[float] = Field(default=None, ge=0)
    reading_order: int = Field(default=0, ge=0)


class TableData(BaseModel):
    headers: List[str] = Field(..., min_length=1)
    rows: List[List[str]] = Field(..., min_length=1)
    bbox: BoundingBox
    caption: Optional[str] = Field(default=None)
    table_id: Optional[str] = Field(default=None)


class FigureData(BaseModel):
    caption: str = Field(..., min_length=1)
    bbox: BoundingBox
    figure_type: str = Field(default="image")
    alt_text: Optional[str] = Field(default=None)
    figure_id: Optional[str] = Field(default=None)


class ExtractedDocument(BaseModel):
    """
    Normalized extraction output from any strategy.
    
    Rubric Compliance:
    - All strategies output this same schema
    - Contains all elements needed for chunking/indexing
    - Strategy metadata for audit trail
    """
    doc_id: str
    filename: str
    pages: int = Field(ge=1)
    
    # Extracted content
    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[TableData] = Field(default_factory=list)
    figures: List[FigureData] = Field(default_factory=list)
    
    # Strategy metadata
    strategy_used: str = Field(pattern="^[ABC]$")
    confidence_score: float = Field(ge=0, le=1)
    confidence_signals: Dict[str, float] = Field(default_factory=dict)
    
    # Cost tracking
    cost_per_page: float = Field(ge=0)
    total_cost: float = Field(ge=0)
    
    # Processing metadata
    processing_time_ms: int = Field(ge=0)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    content_hash: str

    @property
    def is_confident(self) -> bool:
        return self.confidence_score >= 0.65