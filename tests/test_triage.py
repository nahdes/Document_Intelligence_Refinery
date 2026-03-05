"""
tests/test_triage.py
Unit tests for TriageAgent classification logic and ExtractionRouter
confidence-gated escalation. All run without real PDFs.
"""
from __future__ import annotations
import json, uuid
from pathlib import Path
from unittest.mock import patch
import pytest

from src.agents.triage import TriageAgent, DOMAIN_KEYWORDS
from src.agents.extractor import ExtractionRouter
from src.models.schemas import (
    DomainHint, DocumentProfile, ExtractionStrategy, ExtractedDocument,
    LayoutComplexity, OriginType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def policy_path(tmp_path):
    import yaml
    p = tmp_path / "policies.yaml"
    p.write_text(yaml.dump({
        "max_cost_usd": 5.0, "max_pages": 500, "max_file_size_mb": 100.0,
        "min_confidence": 0.65, "min_language_confidence": 0.60,
        "max_escalation_retries": 2,
        "strategy_a_cost_per_page": 0.0001,
        "strategy_b_cost_per_page": 0.005,
        "strategy_c_cost_per_page": 0.02,
        "pii_redaction_enabled": True, "malware_scan_enabled": True,
        "encryption_at_rest": False,
        "allowed_languages": ["en", "am", "fr", "de"],
        "extraction_thresholds": {
            "fast_text_min_chars_per_page": 100.0,
            "fast_text_max_image_area_ratio": 0.50,
            "strategy_a_confidence_floor": 0.70,
            "strategy_b_confidence_floor": 0.60,
            "strategy_c_confidence_floor": 0.50,
        },
    }))
    return p

@pytest.fixture
def agent(tmp_path, policy_path):
    from src.core.constraint_enforcement import ConstraintEnforcementSystem
    from src.core.security import AuditLedger
    ces = ConstraintEnforcementSystem(
        policy_path=policy_path,
        audit_ledger=AuditLedger(path=tmp_path / "audit.jsonl"),
    )
    return TriageAgent(ces=ces, profiles_dir=tmp_path / "profiles")

@pytest.fixture
def base_profile():
    return DocumentProfile(
        doc_id=str(uuid.uuid4()), filename="test.pdf",
        file_size_bytes=1_048_576, page_count=10, mime_type="application/pdf",
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        domain_hint=DomainHint.FINANCIAL, language="en", language_confidence=0.95,
        recommended_strategy=ExtractionStrategy.FAST, estimated_cost_usd=0.001,
        avg_chars_per_page=400.0, avg_image_area_ratio=0.05, content_hash="a" * 64,
    )

def _make_pdf(tmp_path: Path, content: bytes = b"revenue profit assets") -> Path:
    stream = b"BT /F1 12 Tf 72 720 Td (" + content + b") Tj ET"
    body = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
            b"/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>endobj\n")
    obj4 = f"4 0 obj<</Length {len(stream)}>>stream\n".encode() + stream + b"\nendstream endobj\n"
    xref_pos = len(body) + len(obj4)
    xref = (b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n0000000266 00000 n \n")
    trailer = f"trailer<</Size 5/Root 1 0 R>>startxref\n{xref_pos}\n%%EOF\n".encode()
    path = tmp_path / f"test_{uuid.uuid4().hex[:6]}.pdf"
    path.write_bytes(body + obj4 + xref + trailer)
    return path


# ── Origin classification ─────────────────────────────────────────────────────

class TestOriginClassification:
    def test_high_chars_low_image_native_digital(self, agent):
        assert agent._classify_origin(450.0, 0.05, True) == OriginType.NATIVE_DIGITAL

    def test_low_chars_high_image_scanned(self, agent):
        assert agent._classify_origin(12.0, 0.92, False) == OriginType.SCANNED_IMAGE

    def test_high_chars_high_image_mixed(self, agent):
        assert agent._classify_origin(350.0, 0.45, True) == OriginType.MIXED

    def test_no_fonts_is_scanned_regardless_of_chars(self, agent):
        assert agent._classify_origin(500.0, 0.1, False) == OriginType.SCANNED_IMAGE

    def test_exactly_at_char_threshold_native(self, agent):
        assert agent._classify_origin(100.0, 0.05, True) == OriginType.NATIVE_DIGITAL

    def test_below_char_threshold_and_high_image_scanned(self, agent):
        assert agent._classify_origin(99.9, 0.51, False) == OriginType.SCANNED_IMAGE


# ── Layout classification ─────────────────────────────────────────────────────

class TestLayoutClassification:
    def test_tables_and_figures_mixed(self, agent):
        assert agent._classify_layout(True, True, 0.1, 400.0) == LayoutComplexity.MIXED

    def test_tables_only_table_heavy(self, agent):
        assert agent._classify_layout(True, False, 0.05, 400.0) == LayoutComplexity.TABLE_HEAVY

    def test_figures_only_figure_heavy(self, agent):
        assert agent._classify_layout(False, True, 0.1, 400.0) == LayoutComplexity.FIGURE_HEAVY

    def test_high_image_ratio_figure_heavy(self, agent):
        assert agent._classify_layout(False, False, 0.4, 400.0) == LayoutComplexity.FIGURE_HEAVY

    def test_high_chars_multi_column(self, agent):
        assert agent._classify_layout(False, False, 0.05, 900.0) == LayoutComplexity.MULTI_COLUMN

    def test_clean_simple_single_column(self, agent):
        assert agent._classify_layout(False, False, 0.02, 350.0) == LayoutComplexity.SINGLE_COLUMN

    def test_tables_dominate_over_image_ratio(self, agent):
        assert agent._classify_layout(True, False, 0.45, 200.0) == LayoutComplexity.TABLE_HEAVY


# ── Domain classification ─────────────────────────────────────────────────────

class TestDomainClassification:
    def test_financial_keywords(self, agent):
        text = b"revenue profit balance sheet equity assets liabilities annual report fiscal year"
        assert agent._classify_domain(text) == DomainHint.FINANCIAL

    def test_legal_keywords(self, agent):
        text = b"whereas plaintiff defendant jurisdiction clause agreement indemnification liability"
        assert agent._classify_domain(text) == DomainHint.LEGAL

    def test_technical_keywords(self, agent):
        text = b"algorithm implementation specification protocol api endpoint microservice database"
        assert agent._classify_domain(text) == DomainHint.TECHNICAL

    def test_medical_keywords(self, agent):
        text = b"patient diagnosis treatment clinical dosage symptom pharmacology physician adverse"
        assert agent._classify_domain(text) == DomainHint.MEDICAL

    def test_sparse_keywords_general(self, agent):
        assert agent._classify_domain(b"hello world this is a normal document") == DomainHint.GENERAL

    def test_empty_document_general(self, agent):
        assert agent._classify_domain(b"") == DomainHint.GENERAL

    def test_all_four_domains_covered(self):
        assert {DomainHint.FINANCIAL, DomainHint.LEGAL, DomainHint.TECHNICAL, DomainHint.MEDICAL} \
               == set(DOMAIN_KEYWORDS.keys())

    def test_each_domain_has_ten_or_more_keywords(self):
        for d, kws in DOMAIN_KEYWORDS.items():
            assert len(kws) >= 10, f"{d}: only {len(kws)} keywords"

    def test_dominant_domain_wins_tie(self, agent):
        text = (b"revenue profit equity assets liabilities balance sheet fiscal year "
                b"net income earnings annual report dividend cash flow auditor whereas clause")
        assert agent._classify_domain(text) == DomainHint.FINANCIAL


# ── Strategy recommendation ───────────────────────────────────────────────────

class TestStrategyRecommendation:
    def test_scanned_always_vision(self, agent):
        assert agent._recommend_strategy(
            OriginType.SCANNED_IMAGE, LayoutComplexity.SINGLE_COLUMN, 50.0, 0.85
        ) == ExtractionStrategy.VISION

    def test_native_single_col_high_chars_fast(self, agent):
        assert agent._recommend_strategy(
            OriginType.NATIVE_DIGITAL, LayoutComplexity.SINGLE_COLUMN, 400.0, 0.10
        ) == ExtractionStrategy.FAST

    def test_native_table_heavy_layout(self, agent):
        assert agent._recommend_strategy(
            OriginType.NATIVE_DIGITAL, LayoutComplexity.TABLE_HEAVY, 400.0, 0.10
        ) == ExtractionStrategy.LAYOUT

    def test_native_multi_column_layout(self, agent):
        assert agent._recommend_strategy(
            OriginType.NATIVE_DIGITAL, LayoutComplexity.MULTI_COLUMN, 400.0, 0.10
        ) == ExtractionStrategy.LAYOUT

    def test_mixed_origin_layout(self, agent):
        assert agent._recommend_strategy(
            OriginType.MIXED, LayoutComplexity.SINGLE_COLUMN, 300.0, 0.20
        ) == ExtractionStrategy.LAYOUT

    def test_native_high_image_ratio_layout(self, agent):
        assert agent._recommend_strategy(
            OriginType.NATIVE_DIGITAL, LayoutComplexity.SINGLE_COLUMN, 400.0, 0.60
        ) == ExtractionStrategy.LAYOUT

    def test_native_low_chars_layout(self, agent):
        assert agent._recommend_strategy(
            OriginType.NATIVE_DIGITAL, LayoutComplexity.SINGLE_COLUMN, 80.0, 0.05
        ) == ExtractionStrategy.LAYOUT

    def test_figure_heavy_layout(self, agent):
        assert agent._recommend_strategy(
            OriginType.NATIVE_DIGITAL, LayoutComplexity.FIGURE_HEAVY, 400.0, 0.10
        ) == ExtractionStrategy.LAYOUT


# ── Triage confidence scoring ─────────────────────────────────────────────────

class TestTriageConfidenceScoring:
    def test_high_chars_high_lang_conf_high_score(self, agent):
        assert agent._compute_triage_confidence(500.0, 0.05, 0.95) >= 0.85

    def test_scanned_low_chars_low_score(self, agent):
        assert agent._compute_triage_confidence(15.0, 0.92, 0.65) < 0.60

    def test_always_in_0_1_range(self, agent):
        for args in [(0, 1.0, 0.0), (1000, 0.0, 1.0), (250, 0.5, 0.7), (500, 0.3, 0.9)]:
            s = agent._compute_triage_confidence(*args)
            assert 0.0 <= s <= 1.0

    def test_image_ratio_penalises(self, agent):
        assert (agent._compute_triage_confidence(300.0, 0.05, 0.90) >
                agent._compute_triage_confidence(300.0, 0.80, 0.90))

    def test_lang_conf_contributes_positively(self, agent):
        assert (agent._compute_triage_confidence(300.0, 0.10, 0.95) >
                agent._compute_triage_confidence(300.0, 0.10, 0.50))

    def test_zero_chars_clamps_non_negative(self, agent):
        assert agent._compute_triage_confidence(0.0, 1.0, 0.0) >= 0.0

    def test_perfect_doc_near_1(self, agent):
        assert agent._compute_triage_confidence(2000.0, 0.0, 1.0) >= 0.95


# ── Language detection ────────────────────────────────────────────────────────

class TestLanguageDetection:
    def test_empty_returns_english(self, agent):
        lang, conf = agent._detect_language(b"")
        assert lang == "en"
        assert conf <= 0.80

    def test_ethiopic_is_amharic(self, agent):
        text = ("ሰላም " * 30).encode("utf-8")
        lang, _ = agent._detect_language(text)
        assert lang == "am"

    def test_english_detected(self, agent):
        text = (b"This is a clearly English document about financial reporting "
                b"and annual results. " * 5)
        lang, conf = agent._detect_language(text)
        assert lang == "en"
        assert conf >= 0.65


# ── Profile persistence ───────────────────────────────────────────────────────

class TestProfilePersistence:
    def test_profile_written_to_disk(self, agent, tmp_path):
        pdf = _make_pdf(tmp_path)
        profile = agent.profile(pdf)
        written = list((tmp_path / "profiles").glob("*.json"))
        assert len(written) == 1
        data = json.loads(written[0].read_text())
        assert data["doc_id"] == profile.doc_id
        assert data["filename"] == pdf.name

    def test_all_required_fields_present(self, agent, tmp_path):
        pdf = _make_pdf(tmp_path)
        profile = agent.profile(pdf)
        for f in ("doc_id", "filename", "page_count", "origin_type", "layout_complexity",
                  "domain_hint", "recommended_strategy", "triage_confidence", "content_hash"):
            assert getattr(profile, f) is not None

    def test_content_hash_is_sha256_hex(self, agent, tmp_path):
        pdf = _make_pdf(tmp_path)
        profile = agent.profile(pdf)
        assert len(profile.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in profile.content_hash)

    def test_financial_domain_detected_from_pdf_content(self, agent, tmp_path):
        text = b"revenue profit balance sheet equity assets liabilities annual report fiscal year"
        pdf = _make_pdf(tmp_path, text)
        profile = agent.profile(pdf)
        assert profile.domain_hint == DomainHint.FINANCIAL


# ── ExtractionRouter confidence tests ────────────────────────────────────────

class TestExtractionConfidenceScoring:

    def _router(self, tmp_path, policy_path):
        from src.core.constraint_enforcement import ConstraintEnforcementSystem
        from src.core.security import AuditLedger
        ces = ConstraintEnforcementSystem(
            policy_path=policy_path,
            audit_ledger=AuditLedger(path=tmp_path / "audit.jsonl"),
        )
        return ExtractionRouter(ces=ces)

    def test_high_confidence_no_escalation(self, tmp_path, policy_path, base_profile):
        router = self._router(tmp_path, policy_path)
        good_doc = ExtractedDocument(
            doc_id=base_profile.doc_id, source_profile=base_profile,
            strategy_used=ExtractionStrategy.FAST,
            confidence_score=0.92, cost_estimate_usd=0.001,
        )
        with patch.object(router._strategies[ExtractionStrategy.FAST], "extract", return_value=good_doc):
            fake = tmp_path / "fake.pdf"
            fake.write_bytes(b"%PDF-1.4 fake")
            result = router.extract(fake, base_profile)
        assert result.escalation_count == 0
        assert result.confidence_score >= 0.65

    def test_low_confidence_triggers_escalation(self, tmp_path, policy_path):
        profile = DocumentProfile(
            doc_id=str(uuid.uuid4()), filename="lc.pdf",
            file_size_bytes=512_000, page_count=5, mime_type="application/pdf",
            origin_type=OriginType.NATIVE_DIGITAL,
            layout_complexity=LayoutComplexity.SINGLE_COLUMN,
            domain_hint=DomainHint.GENERAL, language="en", language_confidence=0.90,
            recommended_strategy=ExtractionStrategy.FAST,
            estimated_cost_usd=0.0005, content_hash="b" * 64,
        )
        router = self._router(tmp_path, policy_path)
        low_doc = ExtractedDocument(
            doc_id=profile.doc_id, source_profile=profile,
            strategy_used=ExtractionStrategy.FAST,
            confidence_score=0.30, cost_estimate_usd=0.0005,
        )
        high_doc = ExtractedDocument(
            doc_id=profile.doc_id, source_profile=profile,
            strategy_used=ExtractionStrategy.LAYOUT,
            confidence_score=0.84, cost_estimate_usd=0.025,
        )
        with patch.object(router._strategies[ExtractionStrategy.FAST], "extract", return_value=low_doc):
            with patch.object(router._strategies[ExtractionStrategy.LAYOUT], "extract", return_value=high_doc):
                fake = tmp_path / "lc.pdf"
                fake.write_bytes(b"%PDF-1.4 low conf")
                result = router.extract(fake, profile)
        assert result.escalation_count >= 1

    def test_fast_extractor_rewards_dense_text(self):
        from src.strategies.fast_extractor import FastTextExtractor
        ext = FastTextExtractor()
        prof = DocumentProfile(
            doc_id=str(uuid.uuid4()), filename="x.pdf",
            file_size_bytes=1024, page_count=10, mime_type="application/pdf",
            origin_type=OriginType.NATIVE_DIGITAL,
            layout_complexity=LayoutComplexity.SINGLE_COLUMN,
            domain_hint=DomainHint.GENERAL, language="en", language_confidence=0.9,
            recommended_strategy=ExtractionStrategy.FAST, estimated_cost_usd=0.001,
            avg_chars_per_page=300.0, avg_image_area_ratio=0.05, content_hash="c" * 64,
        )
        assert ext._compute_confidence("word " * 3000, prof) > ext._compute_confidence("word " * 100, prof)

    def test_fast_extractor_penalises_image_ratio(self):
        from src.strategies.fast_extractor import FastTextExtractor
        ext = FastTextExtractor()
        mk = lambda img: DocumentProfile(
            doc_id=str(uuid.uuid4()), filename="x.pdf",
            file_size_bytes=1024, page_count=5, mime_type="application/pdf",
            origin_type=OriginType.NATIVE_DIGITAL,
            layout_complexity=LayoutComplexity.SINGLE_COLUMN,
            domain_hint=DomainHint.GENERAL, language="en", language_confidence=0.9,
            recommended_strategy=ExtractionStrategy.FAST, estimated_cost_usd=0.001,
            avg_chars_per_page=300.0, avg_image_area_ratio=img, content_hash="d" * 64,
        )
        text = "word " * 1500
        assert ext._compute_confidence(text, mk(0.05)) > ext._compute_confidence(text, mk(0.80))

    def test_ledger_entry_written_after_extract(self, tmp_path, policy_path, base_profile):
        import src.agents.extractor as ext_mod
        orig = ext_mod.LEDGER_PATH
        ext_mod.LEDGER_PATH = tmp_path / "ledger.jsonl"
        try:
            router = self._router(tmp_path, policy_path)
            good = ExtractedDocument(
                doc_id=base_profile.doc_id, source_profile=base_profile,
                strategy_used=ExtractionStrategy.FAST,
                confidence_score=0.91, cost_estimate_usd=0.001,
            )
            with patch.object(router._strategies[ExtractionStrategy.FAST], "extract", return_value=good):
                fake = tmp_path / "ledger_test.pdf"
                fake.write_bytes(b"%PDF-1.4 ledger test")
                router.extract(fake, base_profile)
            lines = (tmp_path / "ledger.jsonl").read_text().strip().splitlines()
            assert len(lines) >= 1
            entry = json.loads(lines[0])
            assert entry["doc_id"] == base_profile.doc_id
            assert "confidence_score" in entry and "strategy_used" in entry
        finally:
            ext_mod.LEDGER_PATH = orig
