"""Logical Document Unit — Semantic chunk for RAG."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from src.models.common import BoundingBox
import hashlib


class LDU(BaseModel):
    """
    Logical Document Unit — Atomically chunked content.
    
    Rubric Compliance:
    - content_hash for provenance verification
    - page_refs and bounding_box for spatial audit
    - parent_section for hierarchical context
    - chunk_type for specialized handling
    """
    ldu_id: str = Field(..., description="Unique identifier")
    content: str = Field(..., min_length=1, description="Chunk text content")
    chunk_type: Literal["text", "table", "figure", "list", "equation"] = Field(...)
    
    # Spatial provenance (Rubric: Required)
    page_refs: List[int] = Field(..., min_length=1, description="1-indexed pages")
    bounding_box: Optional[BoundingBox] = Field(default=None)
    
    # Hierarchical context (Rubric: Required)
    parent_section: Optional[str] = Field(default=None)
    section_level: int = Field(default=0, ge=0)
    
    # Content metadata
    token_count: int = Field(ge=0)
    content_hash: str = Field(..., description="SHA-256 of content")
    
    # Quality flags
    pii_redacted: bool = Field(default=False)
    is_atomic: bool = Field(default=True, description="True if chunk respects semantic boundaries")
    
    # Relationships
    cross_references: List[str] = Field(default_factory=list)
    related_ldu_ids: List[str] = Field(default_factory=list)

    @classmethod
    def generate_content_hash(cls, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    @classmethod
    def generate_ldu_id(cls, doc_id: str, index: int) -> str:
        return f"{doc_id}_ldu_{index:06d}"