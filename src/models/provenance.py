"""Provenance Chain — Audit trail for extracted claims."""

from pydantic import BaseModel, Field
from typing import List, Optional
from src.models.common import BoundingBox
from datetime import datetime


class Citation(BaseModel):
    """Single source citation with spatial provenance."""
    document_name: str
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    content_hash: str
    excerpt: Optional[str] = Field(default=None, max_length=500)


class ProvenanceChain(BaseModel):
    """Complete provenance for a query answer."""
    answer: str
    confidence: float = Field(ge=0, le=1)
    citations: List[Citation] = Field(default_factory=list)
    
    # Audit flags
    human_review_flag: bool = Field(default=False)
    verification_status: str = Field(default="verified")
    
    # Metadata
    query_id: Optional[str] = Field(default=None)
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    processing_time_ms: int = Field(ge=0)

    @property
    def is_auditable(self) -> bool:
        return len(self.citations) > 0 and self.confidence >= 0.65