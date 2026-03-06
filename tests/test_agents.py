"""
tests/test_agents.py — End-to-end pipeline integration tests.
Full chain: CES → Triage → Extraction → Chunking → Index → Query
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.agents.chunker import ChunkingEngine
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.agents.triage import TriageAgent
from src.core.policy_engine import PolicyViolation
from src.core.security import SecurityViolation
from src.models.schemas import ChunkType, ExtractionStrategy


class TestCESInjection:
    def test_triage_accepts_shared_ces(self, ces, tmp_path):
        triage = TriageAgent(ces=ces, profiles_dir=tmp_path / "profiles")
        assert triage.ces is ces

    def test_router_accepts_shared_ces(self, ces):
        assert ExtractionRouter(ces=ces).ces is ces

    def test_chunker_accepts_shared_ces(self, ces):
        assert ChunkingEngine(ces=ces).ces is ces

    def test_indexer_accepts_shared_ces(self, ces):
        assert PageIndexBuilder(ces=ces).ces is ces

    def test_query_agent_accepts_shared_ces(self, ces, sample_ldus, sample_index):
        assert QueryAgent(ldus=sample_ldus, page_index=sample_index, ces=ces).ces is ces

    def test_all_agents_share_same_audit_ledger(self, ces, sample_ldus, sample_index, tmp_path):
        triage  = TriageAgent(ces=ces, profiles_dir=tmp_path / "profiles")
        router  = ExtractionRouter(ces=ces)
        chunker = ChunkingEngine(ces=ces)
        indexer = PageIndexBuilder(ces=ces)
        agent   = QueryAgent(ldus=sample_ldus, page_index=sample_index, ces=ces)
        assert router.ces is triage.ces
        assert chunker.ces is router.ces
        assert indexer.ces is chunker.ces
        assert agent.ces is indexer.ces


class TestCESGatesCoverage:
    def test_gate_ingest_fires_on_triage(self, ces, minimal_pdf, tmp_path):
        TriageAgent(ces=ces, profiles_dir=tmp_path / "profiles").profile(minimal_pdf)
        events = [e["event"] for e in ces.audit.read_all()]
        assert any("INGEST" in ev for ev in events)

    def test_gate_triage_fires_on_profile(self, ces, minimal_pdf, tmp_path):
        TriageAgent(ces=ces, profiles_dir=tmp_path / "profiles").profile(minimal_pdf)
        events = [e["event"] for e in ces.audit.read_all()]
        assert any("TRIAGE" in ev for ev in events)

    def test_gate_extract_fires(self, ces, sample_profile, minimal_pdf):
        ExtractionRouter(ces=ces).extract(minimal_pdf, sample_profile)
        events = [e["event"] for e in ces.audit.read_all()]
        assert any("EXTRACT" in ev for ev in events)

    def test_gate_confidence_fires(self, ces, sample_profile, minimal_pdf):
        ExtractionRouter(ces=ces).extract(minimal_pdf, sample_profile)
        events = [e["event"] for e in ces.audit.read_all()]
        assert any("CONFIDENCE" in ev for ev in events)

    def test_gate_chunk_fires(self, ces, sample_extracted):
        ChunkingEngine(ces=ces).chunk(sample_extracted)
        events = [e["event"] for e in ces.audit.read_all()]
        assert any("CHUNK" in ev for ev in events)

    def test_gate_index_fires(self, ces, sample_ldus, sample_profile):
        PageIndexBuilder(ces=ces).build(
            doc_id=sample_profile.doc_id, filename=sample_profile.filename,
            ldus=sample_ldus, total_pages=sample_profile.page_count,
        )
        events = [e["event"] for e in ces.audit.read_all()]
        assert any("INDEX" in ev for ev in events)

    def test_gate_query_fires(self, ces, sample_ldus, sample_index):
        QueryAgent(ldus=sample_ldus, page_index=sample_index, ces=ces).query("net profit?")
        events = [e["event"] for e in ces.audit.read_all()]
        assert any("QUERY" in ev for ev in events)


class TestChunkingPipeline:
    def test_produces_ldus(self, ces, sample_extracted):
        assert len(ChunkingEngine(ces=ces).chunk(sample_extracted)) > 0

    def test_table_ldu_is_atomic(self, ces, sample_extracted):
        ldus = ChunkingEngine(ces=ces).chunk(sample_extracted)
        assert any(l.chunk_type == ChunkType.TABLE for l in ldus)

    def test_header_ldus_produced(self, ces, sample_extracted):
        ldus = ChunkingEngine(ces=ces).chunk(sample_extracted)
        assert any(l.chunk_type == ChunkType.HEADER for l in ldus)

    def test_section_propagated_to_children(self, ces, sample_extracted):
        ldus = ChunkingEngine(ces=ces).chunk(sample_extracted)
        header_seen = False
        for ldu in ldus:
            if ldu.chunk_type == ChunkType.HEADER:
                header_seen = True
            elif header_seen and ldu.chunk_type == ChunkType.TEXT:
                assert ldu.parent_section is not None
                break

    def test_all_ldus_have_content_hash(self, ces, sample_extracted):
        for ldu in ChunkingEngine(ces=ces).chunk(sample_extracted):
            assert ldu.content_hash

    def test_no_text_ldu_exceeds_max_tokens(self, ces, sample_extracted):
        max_tokens = ces.policy.chunk_rules.max_tokens
        for ldu in ChunkingEngine(ces=ces).chunk(sample_extracted):
            if ldu.chunk_type not in (ChunkType.TABLE,):
                assert ldu.token_count <= max_tokens + 5


class TestPageIndexPipeline:
    def test_builds_non_empty_index(self, ces, sample_ldus, sample_profile):
        idx = PageIndexBuilder(ces=ces).build(
            doc_id=sample_profile.doc_id, filename=sample_profile.filename,
            ldus=sample_ldus, total_pages=sample_profile.page_count,
        )
        assert len(idx.root_sections) > 0

    def test_sections_have_summaries(self, ces, sample_ldus, sample_profile):
        idx = PageIndexBuilder(ces=ces).build(
            doc_id=sample_profile.doc_id, filename=sample_profile.filename,
            ldus=sample_ldus, total_pages=sample_profile.page_count,
        )
        for sec in idx.root_sections:
            assert sec.summary

    def test_index_persisted_to_disk(self, ces, sample_ldus, sample_profile, tmp_path):
        import src.agents.indexer as mod
        orig = mod.PAGEINDEX_DIR
        mod.PAGEINDEX_DIR = tmp_path / "pageindex"
        mod.PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
        try:
            PageIndexBuilder(ces=ces).build(
                doc_id=sample_profile.doc_id, filename=sample_profile.filename,
                ldus=sample_ldus, total_pages=sample_profile.page_count,
            )
            out = tmp_path / "pageindex" / f"{sample_profile.doc_id}.json"
            assert out.exists()
            assert json.loads(out.read_text())["doc_id"] == sample_profile.doc_id
        finally:
            mod.PAGEINDEX_DIR = orig

    def test_navigate_finds_sections(self, sample_index):
        results = sample_index.navigate("financial highlights profit", top_k=3)
        assert len(results) > 0

    def test_empty_ldus_triggers_gate_warning(self, ces, sample_profile):
        PageIndexBuilder(ces=ces).build(
            doc_id=sample_profile.doc_id, filename=sample_profile.filename,
            ldus=[], total_pages=sample_profile.page_count,
        )
        events = [e["event"] for e in ces.audit.read_all()]
        assert any("EMPTY_INDEX" in ev for ev in events)


class TestQueryAgentPipeline:
    def test_returns_provenance_chain(self, ces, sample_ldus, sample_index):
        chain = QueryAgent(ldus=sample_ldus, page_index=sample_index, ces=ces).query("net profit?")
        assert chain.query
        assert chain.answer
        assert 0.0 <= chain.confidence <= 1.0

    def test_pageindex_tool_for_section_query(self, ces, sample_ldus, sample_index):
        chain = QueryAgent(ldus=sample_ldus, page_index=sample_index, ces=ces).query(
            "Where is the financial highlights section?")
        assert chain.tool_used == "pageindex_navigate"

    def test_semantic_search_tool_by_default(self, ces, sample_ldus, sample_index):
        chain = QueryAgent(ldus=sample_ldus, page_index=sample_index, ces=ces).query(
            "Tell me about capital adequacy")
        assert chain.tool_used == "semantic_search"

    def test_low_confidence_flagged_for_human_review(self, ces, sample_index):
        chain = QueryAgent(ldus=[], page_index=sample_index, ces=ces).query(
            "What is the completely unknown metric xyzzy?")
        assert chain.human_review_required is True
        assert "HUMAN REVIEW" in chain.answer

    def test_audit_claim_returns_status(self, ces, sample_ldus, sample_index):
        result = QueryAgent(ldus=sample_ldus, page_index=sample_index, ces=ces).audit_claim(
            "Commercial Bank of Ethiopia total assets")
        assert result["status"] in ("VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIABLE")

    def test_audit_claim_unverifiable_on_empty(self, ces, sample_index):
        result = QueryAgent(ldus=[], page_index=sample_index, ces=ces).audit_claim("xyzzy nonce claim")
        assert result["status"] == "UNVERIFIABLE"


class TestSecurityIntegration:
    def test_malware_blocked_at_triage(self, ces, eicar_file, tmp_path):
        with pytest.raises(SecurityViolation):
            TriageAgent(ces=ces, profiles_dir=tmp_path / "profiles").profile(eicar_file)

    def test_clean_document_passes_all_gates(self, ces, minimal_pdf, tmp_path):
        profile = TriageAgent(ces=ces, profiles_dir=tmp_path / "profiles").profile(minimal_pdf)
        assert profile.doc_id
        assert profile.page_count >= 1


class TestAuditLedgerIntegrity:
    def test_chain_intact_after_pipeline(self, ces, sample_extracted, sample_ldus, sample_profile):
        ChunkingEngine(ces=ces).chunk(sample_extracted)
        PageIndexBuilder(ces=ces).build(
            doc_id=sample_profile.doc_id, filename=sample_profile.filename,
            ldus=sample_ldus, total_pages=sample_profile.page_count,
        )
        is_valid, msg = ces.audit.verify_chain()
        assert is_valid, f"Audit chain broken: {msg}"

    def test_tamper_detection(self, tmp_path, policy_path):
        from src.core.security import AuditLedger
        ledger = AuditLedger(path=tmp_path / "tamper.jsonl")
        ledger.append("EVENT_1", {"data": "original"})
        ledger.append("EVENT_2", {"data": "second"})
        lines = ledger.path.read_text().splitlines()
        entry = json.loads(lines[0])
        entry["payload"]["data"] = "TAMPERED"
        lines[0] = json.dumps(entry)
        ledger.path.write_text("\n".join(lines) + "\n")
        is_valid, msg = ledger.verify_chain()
        assert not is_valid

    def test_ledger_has_entries_after_query(self, ces, sample_ldus, sample_index):
        QueryAgent(ldus=sample_ldus, page_index=sample_index, ces=ces).query("key metrics?")
        assert len(ces.audit.read_all()) > 0
