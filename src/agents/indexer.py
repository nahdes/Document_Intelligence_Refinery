"""
src/agents/indexer.py
PageIndexBuilder — Stage 4. Builds hierarchical PageIndex from LDUs.
CES gate_index() validates the completed index. Persists to .refinery/pageindex/{doc_id}.json.
"""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Optional

from src.core.constraint_enforcement import ConstraintEnforcementSystem
from src.models.schemas import ChunkType, LDU, PageIndex, PageIndexNode

logger = logging.getLogger(__name__)
PAGEINDEX_DIR = Path(".refinery/pageindex")

ENTITY_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+(?:of|the|and|for|in|on|at|by|to)\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)+\b"
)


class PageIndexBuilder:
    """Builds a hierarchical PageIndex navigation tree over a document's LDUs."""

    def __init__(self, ces: Optional[ConstraintEnforcementSystem] = None) -> None:
        self.ces = ces or ConstraintEnforcementSystem()
        PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)

    def build(self, doc_id: str, filename: str, ldus: list[LDU], total_pages: int) -> PageIndex:
        """Build PageIndex from LDUs, validate via CES gate_index, persist to disk."""
        ldu_map = {ldu.chunk_id: ldu for ldu in ldus}
        root_sections: list[PageIndexNode] = []
        current_node: Optional[PageIndexNode] = None
        depth_stack: list[PageIndexNode] = []

        for ldu in ldus:
            if ldu.chunk_type == ChunkType.HEADER:
                level = self._heading_level(ldu.content)
                node = PageIndexNode(
                    title=ldu.content.strip()[:120],
                    page_start=min(ldu.page_refs) if ldu.page_refs else 1,
                    page_end=max(ldu.page_refs) if ldu.page_refs else 1,
                    level=level, ldu_ids=[ldu.chunk_id],
                )
                if level == 0 or not depth_stack:
                    if current_node and current_node not in root_sections:
                        root_sections.append(current_node)
                    depth_stack = [node]
                    current_node = node
                else:
                    while len(depth_stack) > 1 and depth_stack[-1].level >= level:
                        depth_stack.pop()
                    depth_stack[-1].child_sections.append(node)
                    depth_stack.append(node)
                    current_node = node
            elif current_node is not None:
                current_node.ldu_ids.append(ldu.chunk_id)
                if ldu.page_refs:
                    current_node.page_end = max(current_node.page_end, max(ldu.page_refs))
                if ldu.chunk_type == ChunkType.TABLE and "tables" not in current_node.data_types_present:
                    current_node.data_types_present.append("tables")
                if ldu.chunk_type == ChunkType.FIGURE and "figures" not in current_node.data_types_present:
                    current_node.data_types_present.append("figures")
            else:
                if not root_sections:
                    preamble = PageIndexNode(
                        title="[Document Preamble]", page_start=1,
                        page_end=max(ldu.page_refs) if ldu.page_refs else 1,
                        level=0, ldu_ids=[ldu.chunk_id],
                    )
                    root_sections.append(preamble)
                    current_node = preamble
                else:
                    root_sections[-1].ldu_ids.append(ldu.chunk_id)

        if current_node and current_node not in root_sections:
            root_sections.append(current_node)

        # Summaries + entities
        all_nodes = self._flatten(root_sections)
        for node in all_nodes:
            node_ldus = [ldu_map[cid] for cid in node.ldu_ids if cid in ldu_map]
            node.summary      = self._summary(node_ldus)
            node.key_entities = self._entities(node_ldus)

        index = PageIndex(
            doc_id=doc_id, filename=filename,
            total_pages=total_pages, root_sections=root_sections,
        )

        # Gate 5: validate
        gate = self.ces.gate_index(doc_id=doc_id, section_count=len(all_nodes), total_ldu_count=len(ldus))
        if not gate.passed:
            logger.warning("PageIndexBuilder gate_index warning: %s", gate.violation_detail)

        out = PAGEINDEX_DIR / f"{doc_id}.json"
        out.write_text(index.model_dump_json(indent=2), encoding="utf-8")

        logger.info(
            "PageIndexBuilder: %s → %d root sections, %d total nodes, %d LDUs",
            filename, len(root_sections), len(all_nodes), len(ldus),
        )
        return index

    def _heading_level(self, text: str) -> int:
        s = text.strip()
        if s.startswith("###"): return 2
        if s.startswith("##"):  return 1
        if s.startswith("#"):   return 0
        if s.isupper() and len(s) < 80: return 0
        if re.match(r"^\d+\.\d+", s): return 1
        if re.match(r"^\d+\.", s):     return 0
        return 1

    def _flatten(self, sections: list[PageIndexNode]) -> list[PageIndexNode]:
        result = []
        for node in sections:
            result.append(node)
            result.extend(self._flatten(node.child_sections))
        return result

    def _summary(self, ldus: list[LDU]) -> str:
        text = " ".join(
            ldu.content for ldu in ldus[:5]
            if ldu.chunk_type in (ChunkType.TEXT, ChunkType.HEADER)
        )[:1000]
        if not text.strip():
            return "No text content available."
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        s = " ".join(s.strip() for s in sentences[:2] if s.strip())
        return (s + ".") if s and not s.endswith(".") else s

    def _entities(self, ldus: list[LDU]) -> list[str]:
        combined = " ".join(
            ldu.content for ldu in ldus[:8]
            if ldu.chunk_type in (ChunkType.TEXT, ChunkType.HEADER, ChunkType.TABLE)
        )[:3000]
        freq: dict[str, int] = {}
        for e in ENTITY_RE.findall(combined):
            e = e.strip()
            if len(e) > 3:
                freq[e] = freq.get(e, 0) + 1
        return sorted(freq, key=lambda x: freq[x], reverse=True)[:10]