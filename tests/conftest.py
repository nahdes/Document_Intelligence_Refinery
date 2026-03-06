"""tests/conftest.py — Shared pytest fixtures for all test modules."""
from __future__ import annotations
import json, uuid
from pathlib import Path
import pytest
from src.core.constraint_enforcement import ConstraintEnforcementSystem
from src.core.security import AuditLedger
from src.models.schemas import (
    BoundingBox, ChunkType, DocumentProfile, DomainHint, ExtractionStrategy,
    ExtractedDocument, ExtractedTable, LDU, LayoutComplexity, OriginType,
    PageIndex, PageIndexNode, TextBlock,
)

def _write_policy(tmp_path: Path, overrides: dict | None = None) -> Path:
    import yaml
    defaults = {
        "max_cost_usd": 0.50, "max_pages": 100, "max_file_size_mb": 10.0,
        "min_confidence": 0.65, "min_language_confidence": 0.60,
        "max_escalation_retries": 2,
        "strategy_a_cost_per_page": 0.0001, "strategy_b_cost_per_page": 0.005,
        "strategy_c_cost_per_page": 0.02,
        "allowed_languages": ["en", "am", "fr", "de"],
        "pii_redaction_enabled": True, "malware_scan_enabled": True,
        "encryption_at_rest": False,
    }
    if overrides:
        defaults.update(overrides)
    path = tmp_path / "policies.yaml"
    path.write_text(yaml.dump(defaults), encoding="utf-8")
    return path

def _make_minimal_pdf() -> bytes:
    stream = b"BT /F1 12 Tf 72 720 Td (CBE Annual Report 2023. Total assets ETB 1.4 trillion. Net profit ETB 12.5 billion.) Tj ET"
    obj4 = f"4 0 obj<</Length {len(stream)}>>stream\n".encode() + stream + b"\nendstream endobj\n"
    body = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
            b"/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>endobj\n")
    xref_pos = len(body) + len(obj4)
    xref = (b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n0000000266 00000 n \n")
    trailer = f"trailer<</Size 5/Root 1 0 R>>startxref\n{xref_pos}\n%%EOF\n".encode()
    return body + obj4 + xref + trailer

@pytest.fixture
def policy_path(tmp_path):
    return _write_policy(tmp_path)

@pytest.fixture
def ces(policy_path, tmp_path):
    ledger = AuditLedger(path=tmp_path / "audit_ledger.jsonl")
    return ConstraintEnforcementSystem(policy_path=policy_path, audit_ledger=ledger)

@pytest.fixture
def ces_strict(tmp_path):
    policy_path = _write_policy(tmp_path, {"max_pages": 5, "max_cost_usd": 0.01, "min_confidence": 0.80})
    return ConstraintEnforcementSystem(policy_path=policy_path, audit_ledger=AuditLedger(tmp_path / "al.jsonl"))

@pytest.fixture
def minimal_pdf(tmp_path):
    path = tmp_path / "test.pdf"
    path.write_bytes(_make_minimal_pdf())
    return path

@pytest.fixture
def eicar_file(tmp_path):
    path = tmp_path / "eicar.pdf"
    path.write_bytes(b"%PDF-1.4 X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*")
    return path

@pytest.fixture
def pdf_magic_bytes():
    return b"%PDF-1.4 sample content for testing. No malware here."

@pytest.fixture
def eicar_bytes():
    return b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"

@pytest.fixture
def audit_ledger(tmp_path):
    return AuditLedger(path=tmp_path / "test_audit.jsonl")

@pytest.fixture
def sample_profile():
    return DocumentProfile(
        doc_id=str(uuid.uuid4()), filename="cbe_annual_report_2023.pdf",
        file_size_bytes=2*1024*1024, page_count=48, mime_type="application/pdf",
        origin_type=OriginType.NATIVE_DIGITAL, layout_complexity=LayoutComplexity.TABLE_HEAVY,
        domain_hint=DomainHint.FINANCIAL, language="en", language_confidence=0.92,
        recommended_strategy=ExtractionStrategy.LAYOUT, estimated_cost_usd=0.24,
        triage_confidence=0.87, avg_chars_per_page=420.0, avg_image_area_ratio=0.12,
        has_embedded_fonts=True, has_tables=True, has_figures=True, content_hash="abc" + "0"*61,
    )

@pytest.fixture
def sample_extracted(sample_profile):
    return ExtractedDocument(
        doc_id=sample_profile.doc_id, source_profile=sample_profile,
        strategy_used=ExtractionStrategy.LAYOUT,
        text_blocks=[
            TextBlock(text="COMMERCIAL BANK OF ETHIOPIA", page=1, is_header=True, font_size=18.0, reading_order=0),
            TextBlock(text="Annual Report 2022/2023", page=1, is_header=True, font_size=14.0, reading_order=1),
            TextBlock(text="The Commercial Bank of Ethiopia achieved record performance in FY2022/23. Total assets grew by 28.4% to ETB 1.4 trillion. Net profit increased 22.1% to ETB 12.5 billion.", page=2, reading_order=2),
            TextBlock(text="Financial Highlights", page=4, is_header=True, font_size=14.0, reading_order=3),
            TextBlock(text="Capital Adequacy Ratio was 14.8%, above the 8% minimum. NPL ratio improved to 3.2%.", page=3, reading_order=4),
        ],
        tables=[ExtractedTable(
            headers=["Indicator", "2022/23", "2021/22", "Change (%)"],
            rows=[["Total Assets (ETB Bn)", "1,412.3", "1,099.5", "+28.4%"],
                  ["Net Profit (ETB Bn)", "12.5", "10.2", "+22.5%"],
                  ["NPL Ratio (%)", "3.2%", "4.1%", "-0.9pp"]],
            page=4, caption="Key Financial Indicators", confidence=0.91,
            bbox=BoundingBox(x0=72, y0=200, x1=540, y1=400, page=4),
        )],
        confidence_score=0.88, cost_estimate_usd=0.24, processing_time_s=4.2,
    )

@pytest.fixture
def sample_ldus(sample_extracted):
    doc_id = sample_extracted.doc_id
    return [
        LDU(doc_id=doc_id, chunk_type=ChunkType.HEADER, content="COMMERCIAL BANK OF ETHIOPIA", token_count=4, page_refs=[1]),
        LDU(doc_id=doc_id, chunk_type=ChunkType.TEXT, content="The Commercial Bank of Ethiopia achieved record performance. Total assets ETB 1.4 trillion. Net profit ETB 12.5 billion.", token_count=22, page_refs=[2], parent_section="COMMERCIAL BANK OF ETHIOPIA"),
        LDU(doc_id=doc_id, chunk_type=ChunkType.HEADER, content="Financial Highlights", token_count=2, page_refs=[4]),
        LDU(doc_id=doc_id, chunk_type=ChunkType.TABLE, content="| Indicator | 2022/23 | 2021/22 |\n| Total Assets | 1,412.3 | 1,099.5 |\n| Net Profit | 12.5 | 10.2 |", token_count=30, page_refs=[4], parent_section="Financial Highlights", table_data={"headers": ["Indicator", "2022/23", "2021/22"], "rows": [["Total Assets (ETB Bn)", "1,412.3", "1,099.5"], ["Net Profit (ETB Bn)", "12.5", "10.2"]]}),
        LDU(doc_id=doc_id, chunk_type=ChunkType.TEXT, content="Capital Adequacy Ratio 14.8%, above 8% minimum. NPL ratio 3.2%.", token_count=14, page_refs=[3], parent_section="Financial Highlights"),
    ]

@pytest.fixture
def sample_index(sample_ldus, sample_profile):
    doc_id = sample_profile.doc_id
    return PageIndex(
        doc_id=doc_id, filename=sample_profile.filename, total_pages=sample_profile.page_count,
        root_sections=[
            PageIndexNode(title="COMMERCIAL BANK OF ETHIOPIA", page_start=1, page_end=3, level=0,
                summary="CBE achieved record performance in FY2022/23 with total assets of ETB 1.4 trillion.",
                key_entities=["Commercial Bank of Ethiopia", "CBE"], ldu_ids=[sample_ldus[0].chunk_id, sample_ldus[1].chunk_id]),
            PageIndexNode(title="Financial Highlights", page_start=4, page_end=4, level=0,
                summary="Key indicators: 28.4% asset growth, 22.5% profit increase, NPL 3.2%.",
                key_entities=["Total Assets", "Net Profit", "Capital Adequacy"], data_types_present=["tables"],
                ldu_ids=[sample_ldus[2].chunk_id, sample_ldus[3].chunk_id, sample_ldus[4].chunk_id]),
        ],
    )
