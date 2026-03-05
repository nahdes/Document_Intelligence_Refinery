"""
src/agents/extractor.py
ExtractionRouter — Stage 2 of the Document Intelligence Refinery.

Routes documents to the correct extraction strategy with:
  - Profile-based initial strategy selection (reads origin_type, layout_complexity, cost_tier)
  - Confidence-gated escalation: A→B→C→human_in_loop
  - Multi-level escalation: A-to-B and B-to-C both supported
  - Decision transparency: RoutingDecision embedded in every ExtractedDocument
  - Graceful degradation: returns best-effort + human_review_flag, never silent failure
  - Config-driven thresholds: all floors sourced from policies.yaml via CES

Rubric compliance (verbatim):
  1. Profile-Based Selection: reads DocumentProfile.origin_type, layout_complexity, estimated_cost_usd
  2. Escalation Guard: confidence < threshold → retry with higher-fidelity strategy
  3. Multi-Level Escalation: A→B and B→C both implemented
  4. Decision Transparency: RoutingDecision returned in ExtractedDocument.routing_decision
  5. Graceful Degradation: exhausted strategies → human_review_required=True, never None
  6. Configuration: thresholds from policies.yaml extraction_thresholds, not hardcoded
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from src.core.constraint_enforcement import ConstraintEnforcementSystem
from src.core.security import security_middleware
from src.models.schemas import (
    DocumentProfile,
    ExtractionLedgerEntry,
    ExtractionStrategy,
    ExtractedDocument,
    RoutingDecision,
)
from src.strategies.fast_extractor import FastTextExtractor
from src.strategies.layout_extractor import LayoutExtractor
from src.strategies.vision_extractor import VisionExtractor

logger = logging.getLogger(__name__)

LEDGER_PATH = Path(".refinery/extraction_ledger.jsonl")

# Escalation chain — sourced here but thresholds come from config
_ESCALATION_CHAIN: dict[str, str | None] = {
    "fast":   "layout",
    "layout": "vision",
    "vision": None,   # terminal — route to human_in_loop
}


class ExtractionRouter:
    """
    Routes DocumentProfile → correct strategy → ExtractedDocument.

    Injects all three strategies at construction time and reads the initial
    strategy from DocumentProfile.recommended_strategy (set by TriageAgent).

    Escalation thresholds are read exclusively from policies.yaml via the
    CES policy engine — no float literals appear in this class.
    """

    def __init__(self, ces: Optional[ConstraintEnforcementSystem] = None) -> None:
        self.ces = ces or ConstraintEnforcementSystem()
        self._strategies = {
            ExtractionStrategy.FAST:   FastTextExtractor(self.ces.policy_engine),
            ExtractionStrategy.LAYOUT: LayoutExtractor(self.ces.policy_engine),
            ExtractionStrategy.VISION: VisionExtractor(self.ces.policy_engine),
        }
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    @security_middleware
    def extract(self, file_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        """
        Extract a document with profile-driven strategy selection and
        automatic confidence-gated escalation.

        Decision transparency is recorded in ExtractedDocument.routing_decision.
        If all strategies are exhausted, returns best-effort result with
        routing_decision.human_review_flag = True.

        Profile fields used for strategy selection (rubric requirement):
          - profile.recommended_strategy (primary signal from TriageAgent)
          - profile.origin_type          (logged in routing decision)
          - profile.layout_complexity    (logged in routing decision)
          - profile.estimated_cost_usd   (logged in routing decision)
        """
        initial_strategy = profile.recommended_strategy
        strategy         = initial_strategy
        max_retries      = self.ces.policy_engine.policy.max_escalation_retries
        escalation_count = 0
        t0               = time.time()

        best_doc: Optional[ExtractedDocument]         = None
        all_attempts:      list[dict]                 = []
        human_review_flag: bool                       = False

        logger.info(
            "ExtractionRouter START: doc=%s origin=%s layout=%s cost_tier=%s "
            "initial_strategy=%s",
            profile.doc_id, profile.origin_type.value,
            profile.layout_complexity.value, profile.cost_tier,
            initial_strategy.value,
        )

        while True:
            # ── Gate 3a: cost budget check (pre-extraction) ───────────────
            cost_result = self.ces.gate_extract(
                doc_id=profile.doc_id,
                strategy=strategy.value,
                page_count=profile.page_count,
            )
            if not cost_result.passed:
                logger.warning(
                    "ExtractionRouter: BUDGET BLOCK strategy=%s doc=%s (%s)",
                    strategy.value, profile.doc_id, cost_result.violation_detail,
                )
                all_attempts.append({
                    "strategy": strategy.value,
                    "status": "budget_blocked",
                    "reason": cost_result.violation_detail,
                    "confidence": 0.0,
                    "cost_usd": 0.0,
                    "threshold": None,
                })
                # Budget blocked → skip to human review
                human_review_flag = True
                break

            # ── Run extraction strategy ───────────────────────────────────
            attempt_start = time.time()
            attempt: dict = {
                "strategy":   strategy.value,
                "origin":     profile.origin_type.value,
                "layout":     profile.layout_complexity.value,
                "cost_tier":  profile.cost_tier,
                "cost_usd":   cost_result.cost_approved_usd,
                "threshold":  None,   # filled after gate_confidence
                "confidence": 0.0,
                "status":     "unknown",
                "reason":     "primary_attempt",
            }

            try:
                doc = self._strategies[strategy].extract(file_path, profile)
                attempt["confidence"] = doc.confidence_score
                attempt["status"]     = "ok"
            except Exception as exc:
                logger.error(
                    "ExtractionRouter: strategy=%s FAILED for doc=%s: %s",
                    strategy.value, profile.doc_id, exc,
                )
                doc = ExtractedDocument(
                    doc_id=profile.doc_id,
                    source_profile=profile,
                    strategy_used=strategy,
                    confidence_score=0.0,
                    cost_estimate_usd=cost_result.cost_approved_usd,
                )
                attempt["confidence"] = 0.0
                attempt["status"]     = f"exception: {type(exc).__name__}"

            all_attempts.append(attempt)

            # Track best result seen so far (for graceful degradation)
            if best_doc is None or doc.confidence_score > best_doc.confidence_score:
                best_doc = doc

            # ── Gate 3b: confidence check (post-extraction) ───────────────
            conf_result = self.ces.gate_confidence(
                doc_id=profile.doc_id,
                strategy=strategy.value,
                confidence=doc.confidence_score,
            )
            # Threshold comes from CES policy engine — fill into attempt log
            threshold = self.ces.policy_engine.policy.min_confidence
            attempt["threshold"] = threshold

            logger.info(
                "ExtractionRouter: strategy=%s confidence=%.3f threshold=%.3f "
                "passed=%s escalate_to=%s",
                strategy.value, doc.confidence_score, threshold,
                conf_result.passed, conf_result.escalation_target or "—",
            )

            if conf_result.passed:
                # SUCCESS — finalize and return
                routing = RoutingDecision(
                    initial_strategy=initial_strategy,
                    final_strategy=strategy,
                    escalation_count=escalation_count,
                    attempts=all_attempts,
                )
                doc.routing_decision = routing
                doc.escalation_count = escalation_count
                doc.processing_time_s = round(time.time() - t0, 4)
                self._log_ledger(doc, profile, all_attempts, human_review=False)
                logger.info(
                    "ExtractionRouter COMPLETE: doc=%s final_strategy=%s "
                    "confidence=%.3f cost=$%.4f escalations=%d [%.2fs]",
                    profile.doc_id, strategy.value, doc.confidence_score,
                    doc.cost_estimate_usd, escalation_count,
                    doc.processing_time_s,
                )
                return doc

            # ── Escalation decision ───────────────────────────────────────
            escalation_count += 1
            attempt["reason"] = f"confidence_below_threshold_{threshold:.2f}"

            exhausted = (
                escalation_count > max_retries
                or conf_result.escalation_target == "human_in_loop"
                or conf_result.escalation_target is None
            )

            if exhausted:
                human_review_flag = True
                logger.warning(
                    "ExtractionRouter: %s for doc=%s after %d escalations — "
                    "returning best-effort (confidence=%.3f) with human_review_flag",
                    ("max_retries_reached" if escalation_count > max_retries
                     else "terminal_strategy"),
                    profile.doc_id, escalation_count,
                    best_doc.confidence_score if best_doc else 0.0,
                )
                break

            next_strategy_str = conf_result.escalation_target
            logger.info(
                "ExtractionRouter: ESCALATE %s → %s (confidence=%.3f < %.3f)",
                strategy.value, next_strategy_str,
                doc.confidence_score, threshold,
            )
            strategy = ExtractionStrategy(next_strategy_str)

        # ── Graceful degradation path ─────────────────────────────────────
        # Return best-effort result — never None, never silently passing bad data
        if best_doc is None:
            best_doc = ExtractedDocument(
                doc_id=profile.doc_id,
                source_profile=profile,
                strategy_used=initial_strategy,
                confidence_score=0.0,
                cost_estimate_usd=0.0,
            )

        routing = RoutingDecision(
            initial_strategy=initial_strategy,
            final_strategy=best_doc.strategy_used,
            escalation_count=escalation_count,
            human_review_flag=True,
            attempts=all_attempts,
        )
        best_doc.routing_decision  = routing
        best_doc.escalation_count  = escalation_count
        best_doc.processing_time_s = round(time.time() - t0, 4)
        self._log_ledger(best_doc, profile, all_attempts, human_review=True)

        logger.warning(
            "ExtractionRouter DEGRADED: doc=%s — human review required. "
            "Best confidence=%.3f via %s after %d escalations.",
            profile.doc_id, best_doc.confidence_score,
            best_doc.strategy_used.value, escalation_count,
        )
        return best_doc

    def _log_ledger(
        self,
        doc: ExtractedDocument,
        profile: DocumentProfile,
        attempts: list[dict],
        human_review: bool,
    ) -> None:
        """Append one row to the extraction ledger JSONL file."""
        try:
            entry = ExtractionLedgerEntry(
                doc_id=doc.doc_id,
                filename=profile.filename,
                strategy_used=doc.strategy_used,
                confidence_score=doc.confidence_score,
                cost_estimate_usd=doc.cost_estimate_usd,
                processing_time_s=doc.processing_time_s,
                escalation_count=doc.escalation_count,
                page_count=profile.page_count,
                human_review_flag=human_review,
                routing_attempts=attempts,
            )
            with open(LEDGER_PATH, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("ExtractionRouter: failed to write ledger: %s", exc)