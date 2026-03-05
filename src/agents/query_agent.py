"""
src/agents/query_agent.py
QueryAgent — Stage 5. LangGraph-style agent with 3 tools + CES gate_query().
Tools: pageindex_navigate, structured_query (SQLite FactTable), semantic_search.
Low-confidence answers flagged [HUMAN REVIEW] and queued.
Every answer returns a full ProvenanceChain with bounding-box citations.
"""
from __future__ import annotations
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from src.core.constraint_enforcement import ConstraintEnforcementSystem
from src.core.security import sanitize_doc_id, security_middleware
from src.models.schemas import (
    ChunkType, LDU, PageIndex, PageIndexNode,
    ProvenanceChain, ProvenanceRecord,
)

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       TEXT NOT NULL, chunk_id TEXT NOT NULL,
    key          TEXT NOT NULL, value TEXT NOT NULL,
    unit         TEXT DEFAULT '', page INTEGER DEFAULT 0,
    content_hash TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
"""

_HUMAN_QUEUE = Path(".refinery/human_loop_queue.jsonl")

TOOL_KEYWORDS = {
    "pageindex_navigate": ["where", "section", "chapter", "part", "find section", "locate", "navigate"],
    "structured_query":   ["total", "amount", "sum", "how much", "value", "revenue", "profit",
                           "percentage", "rate", "cost", "balance", "net", "gross", "ratio"],
}


class QueryAgent:
    """Document QA agent with three tools, CES-enforced confidence gate, and full provenance."""

    def __init__(
        self,
        ldus: list[LDU],
        page_index: PageIndex,
        ces: Optional[ConstraintEnforcementSystem] = None,
        db_path: Path = Path(".refinery/fact_table.db"),
    ) -> None:
        self.ldu_map   = {ldu.chunk_id: ldu for ldu in ldus}
        self.ldus_list = ldus
        self.index     = page_index
        self.ces       = ces or ConstraintEnforcementSystem()
        self.db_path   = db_path
        _HUMAN_QUEUE.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._populate_facts()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.executescript(_DDL)
        conn.commit()
        conn.close()

    def _populate_facts(self) -> None:
        conn = sqlite3.connect(self.db_path)
        imported = 0
        for ldu in self.ldus_list:
            if ldu.chunk_type != ChunkType.TABLE or not ldu.table_data:
                continue
            headers = ldu.table_data.get("headers", [])
            rows    = ldu.table_data.get("rows", [])
            page    = ldu.page_refs[0] if ldu.page_refs else 0
            for row in rows:
                if len(row) < 2:
                    continue
                key = str(row[0]).strip()
                for i, cell in enumerate(row[1:], 1):
                    if not cell or not str(cell).strip():
                        continue
                    col   = headers[i] if i < len(headers) else f"col_{i}"
                    fkey  = f"{key} — {col}"
                    exists = conn.execute(
                        "SELECT 1 FROM facts WHERE doc_id=? AND key=? AND value=?",
                        (ldu.doc_id, fkey, cell),
                    ).fetchone()
                    if not exists:
                        conn.execute(
                            "INSERT INTO facts(doc_id,chunk_id,key,value,page,content_hash) VALUES(?,?,?,?,?,?)",
                            (ldu.doc_id, ldu.chunk_id, fkey, str(cell), page, ldu.content_hash),
                        )
                        imported += 1
        conn.commit()
        conn.close()
        if imported:
            logger.info("QueryAgent: %d facts imported from TABLE LDUs", imported)

    @security_middleware
    def query(self, question: str) -> ProvenanceChain:
        """Answer a question using the best available tool, with CES confidence gate."""
        t0   = time.time()
        tool = self._select_tool(question)
        logger.info("QueryAgent: %r → tool=%s", question[:80], tool)

        if tool == "pageindex_navigate":
            sources = self._pageindex_navigate(question)
        elif tool == "structured_query":
            sources = self._structured_query(question)
        else:
            sources = self._semantic_search(question)

        answer, confidence = self._synthesise(question, sources, tool)

        # Gate 6: confidence enforcement
        gate = self.ces.gate_query(doc_id=self.index.doc_id, query=question, confidence=confidence)
        human_review = False
        if not gate.passed:
            answer = f"[LOW CONFIDENCE — HUMAN REVIEW REQUIRED] {answer}"
            human_review = True
            self._queue_human(question, answer, confidence, sources)

        logger.info(
            "QueryAgent: DONE tool=%s sources=%d confidence=%.3f human_review=%s [%.3fs]",
            tool, len(sources), confidence, human_review, round(time.time() - t0, 4),
        )
        return ProvenanceChain(
            query=question, answer=answer, confidence=confidence,
            sources=sources, tool_used=tool, human_review_required=human_review,
        )

    def _select_tool(self, question: str) -> str:
        q = question.lower()
        for kw in TOOL_KEYWORDS["pageindex_navigate"]:
            if kw in q:
                return "pageindex_navigate"
        for kw in TOOL_KEYWORDS["structured_query"]:
            if kw in q:
                if self._fact_table_has_data():
                    return "structured_query"
                break
        return "semantic_search"

    def _fact_table_has_data(self) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            count = conn.execute("SELECT COUNT(*) FROM facts WHERE doc_id=?", (self.index.doc_id,)).fetchone()[0]
            conn.close()
            return count > 0
        except Exception:
            return False

    def _pageindex_navigate(self, question: str) -> list[ProvenanceRecord]:
        nodes = self.index.navigate(question, top_k=3)
        records = []
        for node in nodes:
            for cid in node.ldu_ids[:5]:
                ldu = self.ldu_map.get(cid)
                if ldu and ldu.chunk_type != ChunkType.HEADER:
                    records.append(ProvenanceRecord(
                        doc_id=ldu.doc_id, filename=self.index.filename,
                        page_number=node.page_start, bbox=ldu.bbox,
                        content_hash=ldu.content_hash, chunk_id=ldu.chunk_id,
                        excerpt=f"[Section: {node.title}] {ldu.content[:250]}",
                    ))
                    break
            else:
                if node.ldu_ids:
                    ldu = self.ldu_map.get(node.ldu_ids[0])
                    if ldu:
                        records.append(ProvenanceRecord(
                            doc_id=ldu.doc_id, filename=self.index.filename,
                            page_number=node.page_start, bbox=ldu.bbox,
                            content_hash=ldu.content_hash, chunk_id=ldu.chunk_id,
                            excerpt=node.summary or node.title,
                        ))
        return records

    def _semantic_search(self, question: str) -> list[ProvenanceRecord]:
        stop = {"the","a","an","is","was","are","were","what","how","when","where",
                "which","who","in","on","at","for","of","and","or","to","by","with"}
        q_words = set(question.lower().split()) - stop
        if not q_words:
            return []
        scored = []
        for ldu in self.ldus_list:
            ldu_words = set(ldu.content.lower().split())
            overlap   = len(q_words & ldu_words)
            if overlap == 0:
                continue
            score = overlap / max(len(q_words), 1)
            if ldu.chunk_type == ChunkType.TABLE:  score *= 1.5
            elif ldu.chunk_type == ChunkType.HEADER: score *= 1.2
            scored.append((score, ldu))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            ProvenanceRecord(
                doc_id=ldu.doc_id, filename=self.index.filename,
                page_number=ldu.page_refs[0] if ldu.page_refs else 0,
                bbox=ldu.bbox, content_hash=ldu.content_hash,
                chunk_id=ldu.chunk_id, excerpt=ldu.content[:300],
            )
            for _, ldu in scored[:5]
        ]

    def _structured_query(self, question: str) -> list[ProvenanceRecord]:
        words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
                 if w not in {"what","when","where","which","with","that","this",
                               "total","much","many","show","tell","give"}]
        if not words:
            return self._semantic_search(question)
        records, seen = [], set()
        try:
            conn = sqlite3.connect(self.db_path)
            for word in words[:5]:
                rows = conn.execute(
                    "SELECT chunk_id,key,value,unit,page,content_hash FROM facts "
                    "WHERE doc_id=? AND (LOWER(key) LIKE ? OR LOWER(key) LIKE ?) "
                    "ORDER BY page ASC LIMIT 5",
                    (self.index.doc_id, f"%{word}%", f"{word}%"),
                ).fetchall()
                for chunk_id, key, value, unit, page, chash in rows:
                    uid = f"{chunk_id}:{key[:20]}"
                    if uid in seen: continue
                    seen.add(uid)
                    ldu = self.ldu_map.get(chunk_id)
                    unit_s = f" {unit}" if unit else ""
                    records.append(ProvenanceRecord(
                        doc_id=self.index.doc_id, filename=self.index.filename,
                        page_number=page, bbox=ldu.bbox if ldu else None,
                        content_hash=chash or "", chunk_id=chunk_id,
                        excerpt=f"{key}: {value}{unit_s}",
                    ))
            conn.close()
        except Exception as exc:
            logger.warning("FactTable query error: %s — falling back to semantic", exc)
            return self._semantic_search(question)
        return records[:5] if records else self._semantic_search(question)

    def _synthesise(self, question: str, sources: list[ProvenanceRecord], tool: str) -> tuple[str, float]:
        if not sources:
            return "The requested information was not found in the document corpus.", 0.0
        confidence = min(0.5 + len(sources) * 0.10, 0.92)
        if tool == "structured_query":
            confidence = min(confidence + 0.05, 0.95)
        excerpts = [f"[p.{r.page_number}] {r.excerpt[:200]}" for r in sources[:3]]
        answer = (
            f"Based on {len(sources)} source(s) in '{self.index.filename}':\n"
            + "\n\n".join(excerpts)
        )
        return answer, round(confidence, 3)

    def _queue_human(self, question: str, answer: str, confidence: float, sources: list) -> None:
        entry = {
            "doc_id": self.index.doc_id, "filename": self.index.filename,
            "question": question, "answer": answer, "confidence": confidence,
            "source_count": len(sources),
            "queued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(_HUMAN_QUEUE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("QueryAgent: queued for human review: %s", question[:60])

    def audit_claim(self, claim: str) -> dict[str, Any]:
        sources = self._semantic_search(claim)
        if not sources:
            return {"status": "UNVERIFIABLE", "claim": claim, "reason": "No relevant content found.", "sources": []}
        stop = {"the","a","an","is","was","are","of","in","at"}
        key_words = set(claim.lower().split()) - stop
        matching = [s for s in sources
                    if len(key_words & set(s.excerpt.lower().split())) >= max(1, len(key_words) // 2)]
        if len(matching) >= 2: status = "VERIFIED"
        elif len(matching) == 1: status = "PARTIALLY_VERIFIED"
        else: status = "UNVERIFIABLE"
        return {
            "status": status, "claim": claim,
            "sources": [s.model_dump() for s in matching[:3]],
            "reason": f"{len(matching)} of {len(sources)} sources support the claim.",
        }