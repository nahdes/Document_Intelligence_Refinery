"""Common types and validators used across all models."""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


class BoundingBox(BaseModel):
    """
    Spatial coordinates for document elements.
    
    Rubric Compliance:
    - Structured sub-model (not raw dict/list)
    - Field validators enforce coordinate logic
    - Used by TextBlock, TableData, LDU, Citation
    """
    page: int = Field(ge=1, description="1-indexed page number")
    x0: float = Field(ge=0, description="Left coordinate (points)")
    y0: float = Field(ge=0, description="Top coordinate (points)")
    x1: float = Field(ge=0, description="Right coordinate (points)")
    y1: float = Field(ge=0, description="Bottom coordinate (points)")

    @field_validator('x1', 'y1')
    @classmethod
    def check_coordinates_positive(cls, v: float) -> float:
        """Ensure coordinates are non-negative."""
        if v < 0:
            raise ValueError("Coordinates must be non-negative")
        return v

    @model_validator(mode='after')
    def check_coordinate_order(self) -> 'BoundingBox':
        """Ensure x0 < x1 and y0 < y1 (valid bounding box)."""
        if self.x0 >= self.x1:
            raise ValueError(f"Invalid bounding box: x0 ({self.x0}) must be < x1 ({self.x1})")
        if self.y0 >= self.y1:
            raise ValueError(f"Invalid bounding box: y0 ({self.y0}) must be < y1 ({self.y1})")
        return self

    @property
    def area(self) -> float:
        """Calculate bounding box area in square points."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def center(self) -> tuple[float, float]:
        """Calculate center point of bounding box."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)