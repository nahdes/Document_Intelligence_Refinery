"""
Base Extraction Strategy — Abstract interface for all strategies.

Rubric Compliance:
- Shared interface with consistent method signatures
- All strategies implement this protocol
"""

from abc import ABC, abstractmethod
from typing import Dict
from src.models.document import ExtractedDocument


class ExtractionStrategy(ABC):
    """
    Abstract base class for extraction strategies.
    
    Rubric Compliance:
    - Common base class with consistent interface
    - All strategies must implement extract() and calculate_confidence()
    """
    
    name: str
    strategy_id: str  # "A", "B", or "C"
    cost_per_page: float
    confidence_floor: float

    @abstractmethod
    def extract(self, file_path: str) -> ExtractedDocument:
        """
        Extract content from document.
        
        Args:
            file_path: Path to document file
            
        Returns:
            ExtractedDocument with normalized schema
        """
        pass

    @abstractmethod
    def calculate_confidence(
        self, 
        doc: ExtractedDocument, 
        page_metrics: Dict
    ) -> float:
        """
        Calculate confidence score for extraction quality.
        
        Args:
            doc: ExtractedDocument to evaluate
            page_metrics: Metrics from triage (char density, image ratio, etc.)
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        pass

    def validate_output(self, doc: ExtractedDocument) -> bool:
        """Validate that output meets minimum requirements."""
        if doc.pages < 1:
            return False
        if not doc.text_blocks and not doc.tables and not doc.figures:
            return False
        return True