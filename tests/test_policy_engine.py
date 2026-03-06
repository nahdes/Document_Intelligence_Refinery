"""
tests/test_policy_engine.py
Unit tests for RefineryPolicyEngine, all exception types, and decorators.
"""
from __future__ import annotations
import pytest
from pathlib import Path
from src.core.policy_engine import (
    RefineryPolicyEngine, RefineryPolicy, PolicyViolation,
    LowConfidenceError, BudgetExceededError,
    enforce_policy_before, enforce_policy_after,
)


@pytest.fixture
def engine(tmp_path):
    p = tmp_path / "policies.yaml"
    p.write_text(
        "max_cost_usd: 0.10\nmax_pages: 100\nmax_file_size_mb: 10.0\n"
        "min_confidence: 0.65\nstrategy_a_cost_per_page: 0.0001\n"
        "strategy_b_cost_per_page: 0.005\nstrategy_c_cost_per_page: 0.02\n"
        "allowed_languages: [en, am, fr]\n"
    )
    return RefineryPolicyEngine(policy_path=p)


class TestPolicyViolations:
    def test_page_count_violation(self, engine):
        with pytest.raises(PolicyViolation) as exc:
            engine.enforce_page_count(101, doc_id="doc_001")
        assert exc.value.rule == "max_pages"
        assert "101" in str(exc.value)

    def test_page_count_at_limit_passes(self, engine):
        engine.enforce_page_count(100)  # exactly at limit — should not raise

    def test_page_count_below_limit_passes(self, engine):
        engine.enforce_page_count(50)

    def test_file_size_violation(self, engine):
        with pytest.raises(PolicyViolation) as exc:
            engine.enforce_file_size(11 * 1024 * 1024, doc_id="doc_big")
        assert exc.value.rule == "max_file_size_mb"
        assert exc.value.doc_id == "doc_big"

    def test_file_size_ok(self, engine):
        engine.enforce_file_size(5 * 1024 * 1024)

    def test_language_not_in_allowlist(self, engine):
        with pytest.raises(PolicyViolation) as exc:
            engine.enforce_language("ja", 0.99, doc_id="doc_ja")
        assert exc.value.rule == "allowed_languages"

    def test_language_in_allowlist_passes(self, engine):
        engine.enforce_language("en", 0.99)
        engine.enforce_language("am", 0.90)
        engine.enforce_language("fr", 0.85)

    def test_budget_exceeded(self, engine):
        with pytest.raises(BudgetExceededError) as exc:
            engine.enforce_cost(1.00, doc_id="doc_exp")
        assert isinstance(exc.value, PolicyViolation)  # BudgetExceededError is a PolicyViolation
        assert exc.value.cost == 1.00
        assert exc.value.limit == 0.10

    def test_cost_within_budget_passes(self, engine):
        engine.enforce_cost(0.05)


class TestLowConfidenceError:
    def test_below_threshold_raises(self, engine):
        with pytest.raises(LowConfidenceError) as exc:
            engine.enforce_confidence(0.30, stage="FastExtractor")
        assert exc.value.score == 0.30
        assert exc.value.threshold == 0.65
        assert exc.value.stage == "FastExtractor"

    def test_at_threshold_passes(self, engine):
        engine.enforce_confidence(0.65)  # exactly at threshold

    def test_above_threshold_passes(self, engine):
        engine.enforce_confidence(0.90)

    def test_error_str_contains_stage(self, engine):
        with pytest.raises(LowConfidenceError) as exc:
            engine.enforce_confidence(0.50, stage="LayoutExtractor")
        assert "LayoutExtractor" in str(exc.value)


class TestCostEstimation:
    def test_cost_hierarchy_fast_lt_layout_lt_vision(self, engine):
        fast   = engine.estimate_cost("fast",   10)
        layout = engine.estimate_cost("layout", 10)
        vision = engine.estimate_cost("vision", 10)
        assert fast < layout < vision

    def test_cost_scales_linearly_with_pages(self, engine):
        cost_10 = engine.estimate_cost("fast", 10)
        cost_20 = engine.estimate_cost("fast", 20)
        assert cost_20 == pytest.approx(cost_10 * 2)

    def test_unknown_strategy_uses_layout_rate(self, engine):
        cost = engine.estimate_cost("unknown", 10)
        layout = engine.estimate_cost("layout", 10)
        assert cost == pytest.approx(layout)

    def test_pre_extraction_check_returns_cost(self, engine):
        cost = engine.pre_extraction_check("fast", 10, "doc_1")
        assert cost == pytest.approx(0.001)

    def test_pre_extraction_check_raises_over_budget(self, engine):
        with pytest.raises(BudgetExceededError):
            engine.pre_extraction_check("vision", 10000, "doc_big")


class TestChunkValidation:
    def test_oversized_chunk_raises(self, engine):
        with pytest.raises(PolicyViolation) as exc:
            engine.validate_chunk({"chunk_type": "text", "token_count": 9999, "doc_id": "d1"})
        assert "max_tokens" in exc.value.rule

    def test_valid_chunk_passes(self, engine):
        engine.validate_chunk({"chunk_type": "text", "token_count": 200, "doc_id": "d1"})

    def test_zero_token_chunk_passes(self, engine):
        engine.validate_chunk({"chunk_type": "text", "token_count": 0, "doc_id": "d1"})

    def test_chunk_at_exact_limit_passes(self, engine):
        max_tokens = engine.policy.chunk_rules.max_tokens
        engine.validate_chunk({"chunk_type": "text", "token_count": max_tokens, "doc_id": "d1"})


class TestPreIngestionCheck:
    def test_file_too_large_caught_first(self, engine):
        with pytest.raises(PolicyViolation) as exc:
            engine.pre_ingestion_check(
                file_size_bytes=100 * 1024 * 1024,  # 100MB — too large
                page_count=10,
                language="en",
                lang_confidence=0.90,
            )
        assert exc.value.rule == "max_file_size_mb"

    def test_page_count_violation_caught(self, engine):
        with pytest.raises(PolicyViolation) as exc:
            engine.pre_ingestion_check(
                file_size_bytes=1 * 1024 * 1024,
                page_count=999,  # too many
                language="en",
                lang_confidence=0.90,
            )
        assert exc.value.rule == "max_pages"

    def test_language_violation_caught(self, engine):
        with pytest.raises(PolicyViolation) as exc:
            engine.pre_ingestion_check(
                file_size_bytes=1 * 1024 * 1024,
                page_count=10,
                language="ja",  # not in allowlist
                lang_confidence=0.90,
            )
        assert exc.value.rule == "allowed_languages"

    def test_all_valid_passes(self, engine):
        engine.pre_ingestion_check(
            file_size_bytes=2 * 1024 * 1024,
            page_count=50,
            language="en",
            lang_confidence=0.95,
        )


class TestPolicyModelValidation:
    def test_default_policy_satisfies_cost_hierarchy(self):
        p = RefineryPolicy()
        assert p.strategy_a_cost_per_page < p.strategy_b_cost_per_page
        assert p.strategy_b_cost_per_page < p.strategy_c_cost_per_page

    def test_inverted_cost_hierarchy_raises(self):
        with pytest.raises(ValueError, match="Cost hierarchy violated"):
            RefineryPolicy(
                strategy_a_cost_per_page=0.10,   # A > B — violation
                strategy_b_cost_per_page=0.005,
                strategy_c_cost_per_page=0.02,
            )

    def test_missing_policy_file_uses_defaults(self, tmp_path):
        eng = RefineryPolicyEngine(policy_path=tmp_path / "nonexistent.yaml")
        assert eng.policy.max_pages == 500  # default value

    def test_corrupt_yaml_uses_defaults(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("{{{{not valid yaml::::}}")
        eng = RefineryPolicyEngine(policy_path=p)
        assert eng.policy is not None  # falls back to defaults


class TestPolicyReload:
    def test_hot_reload_picks_up_changes(self, tmp_path):
        p = tmp_path / "policies.yaml"
        p.write_text(
            "max_pages: 100\nstrategy_a_cost_per_page: 0.0001\n"
            "strategy_b_cost_per_page: 0.005\nstrategy_c_cost_per_page: 0.02\n"
        )
        eng = RefineryPolicyEngine(policy_path=p)
        assert eng.policy.max_pages == 100
        # Update on disk
        p.write_text(
            "max_pages: 250\nstrategy_a_cost_per_page: 0.0001\n"
            "strategy_b_cost_per_page: 0.005\nstrategy_c_cost_per_page: 0.02\n"
        )
        eng.reload()
        assert eng.policy.max_pages == 250


class TestDecorators:
    """
    Test @enforce_policy_before and @enforce_policy_after decorators.
    Decorators work on class methods that expose `self.ces.policy_engine`.
    We simulate via a simple stub class.
    """

    def test_enforce_after_catches_low_confidence(self, engine):
        class FakeAgent:
            policy = engine

            @enforce_policy_after(check_confidence=True)
            def extract(self) -> object:
                class R:
                    confidence_score = 0.10
                return R()

        agent = FakeAgent()
        with pytest.raises(LowConfidenceError):
            agent.extract()

    def test_enforce_after_passes_high_confidence(self, engine):
        class FakeAgent:
            policy = engine

            @enforce_policy_after(check_confidence=True)
            def extract(self) -> object:
                class R:
                    confidence_score = 0.95
                return R()

        agent = FakeAgent()
        result = agent.extract()
        assert result.confidence_score == 0.95
