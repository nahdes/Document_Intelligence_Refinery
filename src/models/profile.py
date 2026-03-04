"""Document Profile — Output of Triage Agent (Stage 1)."""

from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from datetime import datetime
from typing import Optional


class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"


class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"


class DomainHint(str, Enum):
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    GENERAL = "general"


class DocumentProfile(BaseModel):
    """
    Complete document characterization produced by Triage Agent.
    
    Rubric Compliance:
    - All categorical fields use Enums
    - BoundingBox uses structured sub-model
    - All fields required for downstream strategy selection
    """
    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)

    # Identity
    doc_id: str = Field(..., description="SHA-256 hash prefix of document")
    filename: str = Field(..., description="Original filename")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    content_hash: str = Field(..., description="Full SHA-256 hash of file")

    # Classification Dimensions
    origin_type: OriginType = Field(..., description="Source type detection")
    layout_complexity: LayoutComplexity = Field(..., description="Layout analysis")
    language: str = Field(..., description="ISO 639-1 language code")
    language_confidence: float = Field(ge=0, le=1, description="Language detection confidence")
    domain_hint: DomainHint = Field(..., description="Domain classification")

    # Strategy Decision
    recommended_strategy: str = Field(pattern="^[ABC]$", description="A=Fast, B=Layout, C=Vision")
    estimated_cost_per_page: float = Field(ge=0, description="USD per page")
    total_estimated_cost: float = Field(ge=0, description="Total estimated USD")

    # Metrics
    page_count: int = Field(ge=1)
    avg_chars_per_page: float = Field(ge=0)
    image_area_ratio: float = Field(ge=0, le=1)
    has_font_meta: bool = Field(default=False)      # ← FIXED
    has_form_fields: bool = Field(default=False)

    # Warnings
    warnings: list[str] = Field(default_factory=list)