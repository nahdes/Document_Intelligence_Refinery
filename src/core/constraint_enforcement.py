"""
ConstraintEnforcementSystem — TRP1 Week 3: Document Intelligence Refinery

Single named entry point that aggregates ALL policy enforcement touchpoints
across the 5-stage pipeline. Every stage calls through this system.

Architecture:
  SecurityGate → TriageAgent → ExtractionRouter → ChunkingEngine → PageIndexBuilder → QueryAgent
       ↓               ↓               ↓                 ↓                ↓               ↓
  [CES.ingest]  [CES.triage]   [CES.extract]    [CES.chunk]      [CES.index]    [CES.query]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.core.policy_engine import (
    BudgetExceededError,
    LowConfidenceError,
    PolicyViolation,
    RefineryPolicy,
    RefineryPolicyEngine,
)
from src.core.security import AuditLedger, SecurityGate, SecurityViolation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enforcement result — returned from every CES gate call
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnforcementResult:
    """Outcome of a CES gate check."""
    stage: str
    passed: bool
    doc_id: str = ""
    violation_rule: str = ""
    violation_detail: str = ""
    escalation_required: bool = False
    escalation_target: str = ""
    cost_approved_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def raise_if_blocked(self) -> None:
        """Re-raise the appropriate exception if this result is a hard block."""
        if not self.passed and not self.escalation_required:
            raise PolicyViolation(self.violation_rule, self.violation_detail, self.doc_id)

    def __str__(self) -> str:
        status = "PASS" if self.passed else ("ESCALATE" if self.escalation_required else "BLOCK")
        return (f"[CES:{self.stage}] {status} | doc={self.doc_id!r} "
                f"rule={self.violation_rule!r} detail={self.violation_detail!r}")


# ─────────────────────────────────────────────────────────────────────────────
# ConstraintEnforcementSystem
# ─────────────────────────────────────────────────────────────────────────────

_ESCALATION_CHAIN = {"fast": "layout", "layout": "vision", "vision": None}


class ConstraintEnforcementSystem:
    """
    Aggregates all policy and security enforcement touchpoints into one named
    system. Every stage of the pipeline calls the relevant gate method here
    rather than invoking policy_engine or security directly.

    Instantiate once and inject into all agents:

        ces = ConstraintEnforcementSystem()
        triage = TriageAgent(ces=ces)
        router = ExtractionRouter(ces=ces)
        chunker = ChunkingEngine(ces=ces)
        indexer = PageIndexBuilder(ces=ces)
        agent   = QueryAgent(ces=ces)
    """

    ESCALATION_CHAIN = _ESCALATION_CHAIN

    def __init__(
        self,
        policy_path: Path | None = None,
        audit_ledger: AuditLedger | None = None,
    ) -> None:
        self.policy_engine = RefineryPolicyEngine(policy_path=policy_path)
        self.policy: RefineryPolicy = self.policy_engine.policy
        self.security_gate = SecurityGate()
        self.audit = audit_ledger or self.security_gate.ledger

        logger.info(
            "ConstraintEnforcementSystem initialized | "
            "max_pages=%d max_cost=$%.2f min_confidence=%.2f pii=%s",
            self.policy.max_pages,
            self.policy.max_cost_usd,
            self.policy.min_confidence,
            self.policy.pii_redaction_enabled,
        )

    # ── Gate 1: Ingestion ─────────────────────────────────────────────────────

    def gate_ingest(
        self,
        file_bytes: bytes,
        filename: str,
        doc_id: str = "",
    ) -> EnforcementResult:
        """
        Full ingestion gate: malware scan + file-type validation +
        size check. Called by SecurityGate / TriageAgent before
        any processing begins.

        Raises SecurityViolation immediately on malware.
        Returns EnforcementResult(passed=False) on policy block.
        """
        stage = "INGEST"
        try:
            meta = self.security_gate.ingest(file_bytes, filename, doc_id)
            result = EnforcementResult(stage=stage, passed=True, doc_id=meta["doc_id"])
            self._audit(stage, "PASS", meta["doc_id"], {"filename": filename,
                        "size_bytes": len(file_bytes)})
            return result

        except SecurityViolation as exc:
            self._audit(stage, "SECURITY_BLOCK", doc_id,
                        {"check": exc.check, "detail": exc.detail})
            raise  # always re-raise security violations

        except PolicyViolation as exc:
            result = EnforcementResult(
                stage=stage, passed=False, doc_id=exc.doc_id,
                violation_rule=exc.rule, violation_detail=exc.detail,
            )
            self._audit(stage, "POLICY_BLOCK", exc.doc_id,
                        {"rule": exc.rule, "detail": exc.detail})
            logger.warning("CES INGEST BLOCK: %s", result)
            return result

    # ── Gate 2: Triage ────────────────────────────────────────────────────────

    def gate_triage(
        self,
        doc_id: str,
        file_size_bytes: int,
        page_count: int,
        language: str,
        lang_confidence: float,
    ) -> EnforcementResult:
        """
        Post-classification gate: enforce page count, file size, and
        language policy. Called by TriageAgent after profiling.
        """
        stage = "TRIAGE"
        warnings: list[str] = []
        try:
            self.policy_engine.pre_ingestion_check(
                file_size_bytes, page_count, language, lang_confidence, doc_id
            )
            if lang_confidence < self.policy.min_language_confidence:
                warnings.append(
                    f"Language confidence {lang_confidence:.2f} below threshold "
                    f"{self.policy.min_language_confidence:.2f}"
                )
            result = EnforcementResult(stage=stage, passed=True,
                                       doc_id=doc_id, warnings=warnings)
            self._audit(stage, "PASS", doc_id,
                        {"pages": page_count, "lang": language, "warnings": warnings})
            return result

        except PolicyViolation as exc:
            result = EnforcementResult(
                stage=stage, passed=False, doc_id=exc.doc_id,
                violation_rule=exc.rule, violation_detail=exc.detail,
            )
            self._audit(stage, "POLICY_BLOCK", exc.doc_id,
                        {"rule": exc.rule, "detail": exc.detail})
            logger.warning("CES TRIAGE BLOCK: %s", result)
            return result

    # ── Gate 3: Extraction ────────────────────────────────────────────────────

    def gate_extract(
        self,
        doc_id: str,
        strategy: str,
        page_count: int,
    ) -> EnforcementResult:
        """
        Pre-extraction gate: validate cost budget before a strategy runs.
        Called by ExtractionRouter for each strategy attempt.

        Returns approved cost in result.cost_approved_usd.
        """
        stage = "EXTRACT"
        try:
            cost = self.policy_engine.pre_extraction_check(strategy, page_count, doc_id)
            result = EnforcementResult(
                stage=stage, passed=True, doc_id=doc_id, cost_approved_usd=cost,
            )
            self._audit(stage, "COST_APPROVED", doc_id,
                        {"strategy": strategy, "pages": page_count, "cost_usd": cost})
            return result

        except BudgetExceededError as exc:
            result = EnforcementResult(
                stage=stage, passed=False, doc_id=exc.doc_id,
                violation_rule=exc.rule, violation_detail=exc.detail,
            )
            self._audit(stage, "BUDGET_EXCEEDED", exc.doc_id,
                        {"strategy": strategy, "detail": exc.detail})
            logger.warning("CES EXTRACT BUDGET BLOCK: %s", result)
            return result

    def gate_confidence(
        self,
        doc_id: str,
        strategy: str,
        confidence: float,
    ) -> EnforcementResult:
        """
        Post-extraction confidence gate. If confidence is below threshold,
        returns an escalation result pointing to the next strategy.
        Called by ExtractionRouter after each extraction attempt.
        """
        stage = "CONFIDENCE"
        try:
            self.policy_engine.enforce_confidence(confidence, stage=f"{strategy}_extractor")
            result = EnforcementResult(stage=stage, passed=True, doc_id=doc_id)
            self._audit(stage, "PASS", doc_id,
                        {"strategy": strategy, "confidence": confidence})
            return result

        except LowConfidenceError as exc:
            next_strategy = self.ESCALATION_CHAIN.get(strategy)
            result = EnforcementResult(
                stage=stage,
                passed=False,
                doc_id=doc_id,
                violation_rule="min_confidence",
                violation_detail=f"score={exc.score:.3f} < threshold={exc.threshold:.3f}",
                escalation_required=next_strategy is not None,
                escalation_target=next_strategy or "human_in_loop",
            )
            self._audit(stage, "LOW_CONFIDENCE_ESCALATE", doc_id, {
                "strategy": strategy, "score": exc.score,
                "threshold": exc.threshold, "escalate_to": result.escalation_target,
            })
            logger.warning("CES CONFIDENCE ESCALATE: %s → %s",
                           strategy, result.escalation_target)
            return result

    # ── Gate 4: Chunking ──────────────────────────────────────────────────────

    def gate_chunk(
        self,
        doc_id: str,
        chunk: dict[str, Any],
    ) -> EnforcementResult:
        """
        Per-chunk validation gate. Enforces all 5 chunking constitution rules.
        Called by ChunkValidator for every LDU before it is emitted.
        """
        stage = "CHUNK"
        try:
            self.policy_engine.validate_chunk(chunk)
            return EnforcementResult(stage=stage, passed=True, doc_id=doc_id)

        except PolicyViolation as exc:
            result = EnforcementResult(
                stage=stage, passed=False, doc_id=doc_id,
                violation_rule=exc.rule, violation_detail=exc.detail,
            )
            logger.warning("CES CHUNK VIOLATION: %s", result)
            return result

    # ── Gate 5: PageIndex ─────────────────────────────────────────────────────

    def gate_index(
        self,
        doc_id: str,
        section_count: int,
        total_ldu_count: int,
    ) -> EnforcementResult:
        """
        Post-index gate: verify the index is non-empty and log completion.
        Called by PageIndexBuilder after the index tree is built.
        """
        stage = "INDEX"
        if section_count == 0:
            result = EnforcementResult(
                stage=stage, passed=False, doc_id=doc_id,
                violation_rule="index_empty",
                violation_detail="PageIndex has 0 sections — extraction likely failed",
            )
            self._audit(stage, "EMPTY_INDEX_WARNING", doc_id,
                        {"sections": section_count, "ldus": total_ldu_count})
            logger.warning("CES INDEX WARNING: %s", result)
            return result

        result = EnforcementResult(stage=stage, passed=True, doc_id=doc_id)
        self._audit(stage, "PASS", doc_id,
                    {"sections": section_count, "ldus": total_ldu_count})
        return result

    # ── Gate 6: Query ─────────────────────────────────────────────────────────

    def gate_query(
        self,
        doc_id: str,
        query: str,
        confidence: float,
    ) -> EnforcementResult:
        """
        Pre-answer gate: enforce minimum confidence on query results.
        Low-confidence answers are flagged for human-in-the-loop review.
        Called by QueryAgent before returning a ProvenanceChain.
        """
        stage = "QUERY"
        warnings: list[str] = []

        # Sanitize query
        if len(query) > 2000:
            warnings.append("Query truncated to 2000 characters")
            query = query[:2000]

        if confidence < self.policy.min_confidence:
            result = EnforcementResult(
                stage=stage,
                passed=False,
                doc_id=doc_id,
                violation_rule="min_confidence",
                violation_detail=(
                    f"Query answer confidence {confidence:.3f} < "
                    f"threshold {self.policy.min_confidence:.3f}"
                ),
                escalation_required=True,
                escalation_target="human_in_loop",
                warnings=warnings,
            )
            self._audit(stage, "LOW_CONFIDENCE_HUMAN_LOOP", doc_id,
                        {"confidence": confidence, "query_preview": query[:80]})
            logger.warning("CES QUERY → human-in-the-loop: confidence=%.3f", confidence)
            return result

        result = EnforcementResult(stage=stage, passed=True,
                                   doc_id=doc_id, warnings=warnings)
        self._audit(stage, "PASS", doc_id, {"confidence": confidence})
        return result

    # ── PII Redaction passthrough ─────────────────────────────────────────────

    def redact_pii(self, text: str, doc_id: str = "") -> tuple[str, list[dict]]:
        """
        Redact PII from extracted text using the security gate's redactor.
        Returns (redacted_text, list_of_redaction_records).
        Only active when policy.pii_redaction_enabled is True.
        """
        if not self.policy.pii_redaction_enabled:
            return text, []
        return self.security_gate.redact_text(text, doc_id)

    # ── Status summary ────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return current policy settings as a human-readable dict."""
        p = self.policy
        return {
            "max_pages": p.max_pages,
            "max_cost_usd": p.max_cost_usd,
            "min_confidence": p.min_confidence,
            "pii_redaction": p.pii_redaction_enabled,
            "encryption_at_rest": p.encryption_at_rest,
            "malware_scan": p.malware_scan_enabled,
            "allowed_languages": p.allowed_languages,
            "chunk_rules": {
                "max_tokens": p.chunk_rules.max_tokens,
                "never_split_table_rows": p.chunk_rules.never_split_table_rows,
                "keep_figure_captions": p.chunk_rules.keep_figure_captions_with_parent,
                "propagate_headers": p.chunk_rules.propagate_section_headers,
            },
            "escalation_chain": self.ESCALATION_CHAIN,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _audit(self, stage: str, outcome: str, doc_id: str, payload: dict) -> None:
        try:
            self.audit.append(f"CES_{stage}_{outcome}", {"doc_id": doc_id, **payload})
        except Exception as exc:
            logger.warning("Audit ledger write failed: %s", exc)
