"""
RefineryPolicyEngine — TRP1 Week 3: Document Intelligence Refinery
Enforces extraction policy, cost limits, confidence gates, PII rules, and
language allowlists across every stage of the pipeline.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class PolicyViolation(Exception):
    """Raised when a hard policy rule is violated (document blocked)."""
    def __init__(self, rule: str, detail: str, doc_id: str = "") -> None:
        self.rule = rule
        self.detail = detail
        self.doc_id = doc_id
        super().__init__(f"[PolicyViolation] rule={rule!r} doc={doc_id!r}: {detail}")


class LowConfidenceError(Exception):
    """Raised when extraction confidence falls below min_confidence.
    Triggers automatic escalation to the next extraction strategy."""
    def __init__(self, score: float, threshold: float, stage: str = "") -> None:
        self.score = score
        self.threshold = threshold
        self.stage = stage
        super().__init__(
            f"[LowConfidenceError] stage={stage!r} score={score:.3f} < threshold={threshold:.3f}"
        )


class BudgetExceededError(PolicyViolation):
    """Raised when estimated extraction cost exceeds max_cost."""
    def __init__(self, cost: float, limit: float, doc_id: str = "") -> None:
        super().__init__(
            rule="max_cost",
            detail=f"cost ${cost:.4f} > limit ${limit:.4f}",
            doc_id=doc_id,
        )
        self.cost = cost
        self.limit = limit


# ─────────────────────────────────────────────────────────────────────────────
# Policy schema (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class ChunkRules(BaseModel):
    max_tokens: int = Field(512, ge=64, le=8192)
    min_tokens: int = Field(64, ge=8, le=512)
    never_split_table_rows: bool = True
    keep_figure_captions_with_parent: bool = True
    keep_lists_intact: bool = True
    propagate_section_headers: bool = True
    resolve_cross_references: bool = True


class ExtractionThresholds(BaseModel):
    fast_text_min_chars_per_page: int = Field(100, ge=0)
    fast_text_max_image_area_ratio: float = Field(0.50, ge=0.0, le=1.0)
    strategy_a_confidence_floor: float = Field(0.70, ge=0.0, le=1.0)
    strategy_b_confidence_floor: float = Field(0.60, ge=0.0, le=1.0)
    vlm_fallback_confidence_floor: float = Field(0.50, ge=0.0, le=1.0)


class RefineryPolicy(BaseModel):
    """Complete policy loaded from rubric/policies.yaml."""

    # Cost controls
    max_cost_usd: float = Field(0.50, ge=0.0)
    strategy_a_cost_per_page: float = Field(0.0001)
    strategy_b_cost_per_page: float = Field(0.005)
    strategy_c_cost_per_page: float = Field(0.02)

    # Document limits
    max_pages: int = Field(500, ge=1)
    max_file_size_mb: float = Field(100.0, ge=0.1)

    # Quality gates
    min_confidence: float = Field(0.65, ge=0.0, le=1.0)
    max_escalation_retries: int = Field(2, ge=0)

    # Security
    pii_redaction_enabled: bool = True
    pii_entities: list[str] = Field(default_factory=lambda: [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
        "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "LOCATION", "NRP", "DATE_TIME",
    ])
    malware_scan_enabled: bool = True
    encryption_at_rest: bool = True

    # Language controls
    allowed_languages: list[str] = Field(
        default_factory=lambda: ["en", "am", "fr", "de", "ar", "zh", "es"]
    )
    min_language_confidence: float = Field(0.75, ge=0.0, le=1.0)

    # Chunking constitution
    chunk_rules: ChunkRules = Field(default_factory=ChunkRules)

    # Extraction thresholds
    extraction_thresholds: ExtractionThresholds = Field(
        default_factory=ExtractionThresholds
    )

    # Audit
    audit_ledger_enabled: bool = True
    immutable_hash_algorithm: str = "sha256"

    @model_validator(mode="after")
    def validate_cost_hierarchy(self) -> "RefineryPolicy":
        if not (self.strategy_a_cost_per_page
                < self.strategy_b_cost_per_page
                < self.strategy_c_cost_per_page):
            raise ValueError("Cost hierarchy violated: strategy_a < strategy_b < strategy_c")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Policy engine
# ─────────────────────────────────────────────────────────────────────────────

class RefineryPolicyEngine:
    """Loads policies from YAML and exposes enforcement methods for every stage."""

    _DEFAULT_POLICY_PATH = Path("rubric/policies.yaml")

    def __init__(self, policy_path: Path | None = None) -> None:
        self._policy_path = policy_path or self._DEFAULT_POLICY_PATH
        self.policy = self._load_policy(self._policy_path)
        logger.info("RefineryPolicyEngine loaded from %s", self._policy_path)

    def reload(self) -> None:
        """Hot-reload policy from disk."""
        self.policy = self._load_policy(self._policy_path)
        logger.info("RefineryPolicyEngine reloaded from %s", self._policy_path)

    @staticmethod
    def _load_policy(path: Path) -> RefineryPolicy:
        try:
            if path.exists():
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                return RefineryPolicy(**data)
        except Exception as exc:
            logger.warning("Failed to load policy from %s (%s) — using defaults.", path, exc)
        return RefineryPolicy()

    # ── Enforcement helpers ────────────────────────────────────────────────────

    def enforce_file_size(self, size_bytes: int, doc_id: str = "") -> None:
        size_mb = size_bytes / (1024 ** 2)
        if size_mb > self.policy.max_file_size_mb:
            raise PolicyViolation("max_file_size_mb",
                f"file is {size_mb:.1f} MB, limit is {self.policy.max_file_size_mb:.1f} MB", doc_id)

    def enforce_page_count(self, page_count: int, doc_id: str = "") -> None:
        if page_count > self.policy.max_pages:
            raise PolicyViolation("max_pages",
                f"document has {page_count} pages, limit is {self.policy.max_pages}", doc_id)

    def enforce_language(self, language: str, confidence: float, doc_id: str = "") -> None:
        if confidence < self.policy.min_language_confidence:
            logger.warning("Language detection confidence %.2f below threshold %.2f for %s",
                           confidence, self.policy.min_language_confidence, doc_id)
        if language not in self.policy.allowed_languages:
            raise PolicyViolation("allowed_languages",
                f"language '{language}' not in allowed list {self.policy.allowed_languages}", doc_id)

    def enforce_confidence(self, score: float, stage: str = "") -> None:
        if score < self.policy.min_confidence:
            raise LowConfidenceError(score, self.policy.min_confidence, stage)

    def enforce_cost(self, estimated_cost: float, doc_id: str = "") -> None:
        if estimated_cost > self.policy.max_cost_usd:
            raise BudgetExceededError(estimated_cost, self.policy.max_cost_usd, doc_id)

    def estimate_cost(self, strategy: str, page_count: int) -> float:
        costs = {
            "fast": self.policy.strategy_a_cost_per_page,
            "layout": self.policy.strategy_b_cost_per_page,
            "vision": self.policy.strategy_c_cost_per_page,
        }
        return costs.get(strategy, self.policy.strategy_b_cost_per_page) * page_count

    def validate_chunk(self, chunk: dict[str, Any]) -> None:
        rules = self.policy.chunk_rules
        token_count = chunk.get("token_count", 0)
        if token_count > rules.max_tokens:
            raise PolicyViolation("chunk_rules.max_tokens",
                f"chunk has {token_count} tokens > max {rules.max_tokens}")
        if token_count < rules.min_tokens and chunk.get("chunk_type") not in ("figure", "table"):
            logger.warning("Chunk %s has only %d tokens (min %d)",
                           chunk.get("chunk_id", "?"), token_count, rules.min_tokens)

    def pre_ingestion_check(self, file_size_bytes: int, page_count: int,
                            language: str, lang_confidence: float, doc_id: str = "") -> None:
        self.enforce_file_size(file_size_bytes, doc_id)
        self.enforce_page_count(page_count, doc_id)
        self.enforce_language(language, lang_confidence, doc_id)

    def pre_extraction_check(self, strategy: str, page_count: int, doc_id: str = "") -> float:
        cost = self.estimate_cost(strategy, page_count)
        self.enforce_cost(cost, doc_id)
        return cost


# ─────────────────────────────────────────────────────────────────────────────
# Decorator factories
# ─────────────────────────────────────────────────────────────────────────────

def _get_engine(engine: RefineryPolicyEngine | None) -> RefineryPolicyEngine:
    return engine or RefineryPolicyEngine()


def enforce_policy_before(
    *,
    check_cost: bool = False,
    check_confidence: bool = False,
    strategy_arg: str = "strategy",
    page_count_arg: str = "page_count",
    confidence_arg: str = "confidence",
    engine: RefineryPolicyEngine | None = None,
) -> Callable:
    """Decorator: run policy checks BEFORE the wrapped function executes."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            pe = _get_engine(engine)
            if check_cost:
                strategy = kwargs.get(strategy_arg, "vision")
                pages = int(kwargs.get(page_count_arg, 1))
                doc_id = kwargs.get("doc_id", "")
                pe.pre_extraction_check(strategy, pages, doc_id)
            if check_confidence:
                score = float(kwargs.get(confidence_arg, 1.0))
                pe.enforce_confidence(score, fn.__name__)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def enforce_policy_after(
    *,
    check_confidence: bool = True,
    confidence_result_key: str = "confidence_score",
    engine: RefineryPolicyEngine | None = None,
) -> Callable:
    """Decorator: run policy checks AFTER the wrapped function executes."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            if check_confidence:
                pe = _get_engine(engine)
                if isinstance(result, dict):
                    score = float(result.get(confidence_result_key, 1.0))
                elif hasattr(result, confidence_result_key):
                    score = float(getattr(result, confidence_result_key))
                else:
                    score = 1.0
                try:
                    pe.enforce_confidence(score, fn.__name__)
                except LowConfidenceError:
                    logger.warning(
                        "Post-execution confidence gate failed in %s (score=%.3f). Escalation required.",
                        fn.__name__, score)
                    raise
            return result
        return wrapper
    return decorator