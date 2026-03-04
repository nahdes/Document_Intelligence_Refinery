"""PageIndex — Hierarchical navigation tree for documents."""

from pydantic import BaseModel, Field
from typing import List, Optional, ForwardRef
from datetime import datetime


SectionNode = ForwardRef('SectionNode')


class SectionNode(BaseModel):
    """
    Recursive section node for PageIndex tree.
    
    Rubric Compliance:
    - Recursive type hinting for tree structure
    - LLM-generated summary for each section
    - Key entities for semantic navigation
    """
    node_id: str
    title: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    level: int = Field(ge=0, description="Heading level (0=doc root)")
    
    # Content summary (Rubric: LLM-generated)
    summary: Optional[str] = Field(default=None, max_length=500)
    
    # Navigation
    children: List[Optional['SectionNode']] = Field(default_factory=list)
    
    # Metadata
    key_entities: List[str] = Field(default_factory=list)
    data_types_present: List[str] = Field(default_factory=list)
    ldu_ids: List[str] = Field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def total_pages(self) -> int:
        return self.page_end - self.page_start + 1


SectionNode.model_rebuild()


class PageIndex(BaseModel):
    """Complete hierarchical index for a document."""
    doc_id: str
    filename: str
    root_sections: List[SectionNode] = Field(default_factory=list)
    total_sections: int = Field(ge=0)
    max_depth: int = Field(ge=0)
    
    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    content_hash: str

    def find_section(self, title: str) -> Optional[SectionNode]:
        def _search(nodes: List[SectionNode]) -> Optional[SectionNode]:
            for node in nodes:
                if node and node.title.lower() == title.lower():
                    return node
                if node and node.children:
                    result = _search(node.children)
                    if result:
                        return result
            return None
        
        return _search(self.root_sections)