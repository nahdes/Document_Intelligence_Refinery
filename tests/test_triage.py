"""
tests/test_triage.py
Unit tests for TriageAgent classification logic.
Written to match the ACTUAL implementation in src/agents/triage.py.
All tests run without real PDFs (except TestProfilePersistence).
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest
import yaml

from src.agents.triage import TriageAgent, DOMAIN_KEYWORDS
from src.models.profile import DocumentProfile, OriginType, LayoutComplexity, DomainHint


# ── Config & Fixtures ─────────────────────────────────────────────────────────

CONFIG = {
    "thresholds": {
        "char_density_min": 500,
        "image_area_max": 0.5,
        "min_confidence": 0.65,
        "max_escalation_retries": 2,
    },
    "cost_tiers": {
        "strategy_a": 0.0001,
        "strategy_b": 0.005,
        "strategy_c": 0.020,
    },
    "domain_classification": {
        "keywords": {
            "financial": ["balance sheet", "revenue", "ebitda", "audit",
                          "income statement", "cash flow"],
            "legal":     ["plaintiff", "defendant", "herein", "contract",
                          "liability", "jurisdiction"],
            "technical": ["architecture", "api", "deployment", "specification",
                          "infrastructure", "microservice"],
            "medical":   ["diagnosis", "patient", "prescription", "clinical",
                          "treatment", "protocol"],
        }
    },
    "chunking_rules": {
        "rule_1": "token_count <= 512",
        "rule_2": "never split table rows",
        "rule_3": "keep fig captions with parent",
        "rule_4": "keep lists intact",
        "rule_5": "propagate section headers",
    }
}


@pytest.fixture
def config_path(tmp_path) -> Path:
    p = tmp_path / "extraction_rules.yaml"
    p.write_text(yaml.dump(CONFIG))
    return p


@pytest.fixture
def agent(config_path) -> TriageAgent:
    return TriageAgent(config_path=str(config_path))


def _make_pdf(tmp_path: Path, content: bytes = b"revenue profit assets") -> Path:
    stream = b"BT /F1 12 Tf 72 720 Td (" + content + b") Tj ET"
    body = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica"
        b">>>>>>>>endobj\n"
    )
    obj4 = f"4 0 obj<</Length {len(stream)}>>stream\n".encode() + stream + b"\nendstream endobj\n"
    xref_pos = len(body) + len(obj4)
    xref = (b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n0000000266 00000 n \n")
    trailer = f"trailer<</Size 5/Root 1 0 R>>startxref\n{xref_pos}\n%%EOF\n".encode()
    path = tmp_path / "test.pdf"
    path.write_bytes(body + obj4 + xref + trailer)
    return path


# ── DOMAIN_KEYWORDS export ────────────────────────────────────────────────────

class TestDomainKeywordsExport:
    def test_module_export_exists(self):
        assert DOMAIN_KEYWORDS is not None

    def test_all_four_domains_covered(self):
        keys = {k.value if hasattr(k, "value") else k for k in DOMAIN_KEYWORDS.keys()}
        assert {"financial", "legal", "technical", "medical"} == keys

    def test_each_domain_has_ten_or_more_keywords(self):
        for domain, kws in DOMAIN_KEYWORDS.items():
            assert len(kws) >= 10, f"{domain}: only {len(kws)} keywords"


# ── Origin classification ─────────────────────────────────────────────────────

class TestOriginClassification:
    def test_high_chars_low_image_native_digital(self, agent):
        assert agent._determine_origin_type(600.0, 0.05, True, 6000) == OriginType.NATIVE_DIGITAL

    def test_low_chars_high_image_scanned(self, agent):
        assert agent._determine_origin_type(10.0, 0.92, False, 10) == OriginType.SCANNED_IMAGE

    def test_zero_chars_high_image_scanned(self, agent):
        assert agent._determine_origin_type(0.0, 0.85, False, 0) == OriginType.SCANNED_IMAGE

    def test_zero_chars_low_image_form_fillable(self, agent):
        assert agent._determine_origin_type(0.0, 0.1, False, 0) == OriginType.FORM_FILLABLE

    def test_high_image_ratio_scanned(self, agent):
        assert agent._determine_origin_type(50.0, 0.85, True, 500) == OriginType.SCANNED_IMAGE

    def test_mixed_some_text_some_images(self, agent):
        assert agent._determine_origin_type(200.0, 0.45, True, 2000) == OriginType.MIXED


# ── Layout classification ─────────────────────────────────────────────────────

class TestLayoutClassification:
    def test_high_table_count_table_heavy(self, agent):
        assert agent._determine_layout_complexity([], False, 3.0) == LayoutComplexity.TABLE_HEAVY

    def test_multi_column_detected(self, agent):
        assert agent._determine_layout_complexity([], True, 0.0) == LayoutComplexity.MULTI_COLUMN

    def test_tables_dominate_over_columns(self, agent):
        assert agent._determine_layout_complexity([], True, 5.0) == LayoutComplexity.TABLE_HEAVY

    def test_clean_single_column(self, agent):
        assert agent._determine_layout_complexity([], False, 0.0) == LayoutComplexity.SINGLE_COLUMN

    def test_borderline_table_count_not_table_heavy(self, agent):
        assert agent._determine_layout_complexity([], False, 1.5) == LayoutComplexity.SINGLE_COLUMN


# ── Domain classification ─────────────────────────────────────────────────────

class TestDomainClassification:
    def test_financial_keywords(self, agent):
        assert agent._classify_domain("revenue balance sheet ebitda audit income statement cash flow") == DomainHint.FINANCIAL

    def test_legal_keywords(self, agent):
        assert agent._classify_domain("plaintiff defendant herein contract liability jurisdiction") == DomainHint.LEGAL

    def test_technical_keywords(self, agent):
        assert agent._classify_domain("architecture api deployment specification infrastructure microservice") == DomainHint.TECHNICAL

    def test_medical_keywords(self, agent):
        assert agent._classify_domain("diagnosis patient prescription clinical treatment protocol") == DomainHint.MEDICAL

    def test_sparse_keywords_general(self, agent):
        assert agent._classify_domain("hello world this is a normal document") == DomainHint.GENERAL

    def test_empty_string_general(self, agent):
        assert agent._classify_domain("") == DomainHint.GENERAL

    def test_dominant_domain_wins(self, agent):
        text = "revenue balance sheet ebitda audit income statement cash flow plaintiff"
        assert agent._classify_domain(text) == DomainHint.FINANCIAL


# ── Strategy selection ────────────────────────────────────────────────────────

class TestStrategySelection:
    def test_native_single_column_fast(self, agent):
        strategy, cost = agent._select_strategy(OriginType.NATIVE_DIGITAL, LayoutComplexity.SINGLE_COLUMN)
        assert strategy == "A"
        assert cost == 0.0001

    def test_scanned_always_vision(self, agent):
        strategy, cost = agent._select_strategy(OriginType.SCANNED_IMAGE, LayoutComplexity.SINGLE_COLUMN)
        assert strategy == "C"
        assert cost == 0.020

    def test_native_table_heavy_layout(self, agent):
        strategy, _ = agent._select_strategy(OriginType.NATIVE_DIGITAL, LayoutComplexity.TABLE_HEAVY)
        assert strategy == "B"

    def test_native_multi_column_layout(self, agent):
        strategy, _ = agent._select_strategy(OriginType.NATIVE_DIGITAL, LayoutComplexity.MULTI_COLUMN)
        assert strategy == "B"

    def test_mixed_origin_layout(self, agent):
        strategy, _ = agent._select_strategy(OriginType.MIXED, LayoutComplexity.SINGLE_COLUMN)
        assert strategy == "B"

    def test_cost_per_strategy_correct(self, agent):
        _, cost_a = agent._select_strategy(OriginType.NATIVE_DIGITAL, LayoutComplexity.SINGLE_COLUMN)
        _, cost_b = agent._select_strategy(OriginType.MIXED, LayoutComplexity.SINGLE_COLUMN)
        _, cost_c = agent._select_strategy(OriginType.SCANNED_IMAGE, LayoutComplexity.SINGLE_COLUMN)
        assert cost_a < cost_b < cost_c


# ── Language detection ────────────────────────────────────────────────────────

class TestLanguageDetection:
    def test_short_text_default_english(self, agent):
        lang, conf = agent._detect_language("hi")
        assert lang == "en"
        assert conf == 0.5

    def test_empty_returns_english(self, agent):
        lang, _ = agent._detect_language("")
        assert lang == "en"

    def test_english_detected(self, agent):
        text = "This is a clearly English document about financial reporting and annual results. " * 5
        lang, _ = agent._detect_language(text)
        assert lang == "en"

    def test_confidence_scales_with_text_length(self, agent):
        _, short_conf = agent._detect_language("hello world")
        _, long_conf = agent._detect_language("This is an English document. " * 20)
        assert long_conf >= short_conf


# ── Profile persistence ───────────────────────────────────────────────────────

class TestProfilePersistence:
    def test_profile_written_to_disk(self, agent, tmp_path):
        pdf = _make_pdf(tmp_path)
        profile = agent.analyze(str(pdf))
        output_dir = tmp_path / "profiles"
        agent.save_profile(profile, str(output_dir))
        assert len(list(output_dir.glob("*.json"))) == 1

    def test_saved_json_contains_doc_id(self, agent, tmp_path):
        pdf = _make_pdf(tmp_path)
        profile = agent.analyze(str(pdf))
        output_dir = tmp_path / "profiles"
        agent.save_profile(profile, str(output_dir))
        data = json.loads(list(output_dir.glob("*.json"))[0].read_text())
        assert data["doc_id"] == profile.doc_id

    def test_all_required_fields_present(self, agent, tmp_path):
        pdf = _make_pdf(tmp_path)
        profile = agent.analyze(str(pdf))
        for f in ("doc_id", "filename", "page_count", "origin_type",
                  "layout_complexity", "domain_hint", "recommended_strategy", "content_hash"):
            assert getattr(profile, f) is not None

    def test_content_hash_is_sha256_hex(self, agent, tmp_path):
        pdf = _make_pdf(tmp_path)
        profile = agent.analyze(str(pdf))
        assert len(profile.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in profile.content_hash)

    def test_financial_domain_detected_from_content(self, agent, tmp_path):
        text = b"revenue balance sheet ebitda audit income statement cash flow"
        pdf = _make_pdf(tmp_path, text)
        profile = agent.analyze(str(pdf))
        assert profile.domain_hint == DomainHint.FINANCIAL.value

    def test_page_count_is_one(self, agent, tmp_path):
        pdf = _make_pdf(tmp_path)
        profile = agent.analyze(str(pdf))
        assert profile.page_count == 1

    def test_strategy_is_valid(self, agent, tmp_path):
        pdf = _make_pdf(tmp_path)
        profile = agent.analyze(str(pdf))
        assert profile.recommended_strategy in ("A", "B", "C")

    def test_warnings_is_list(self, agent, tmp_path):
        pdf = _make_pdf(tmp_path)
        profile = agent.analyze(str(pdf))
        assert isinstance(profile.warnings, list)