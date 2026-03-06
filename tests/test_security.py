"""
tests/test_security.py
Unit tests for src/core/security.py — FileTypeValidator, MalwareScanner,
PIIRedactor, AuditLedger, SecurityGate, sanitize_path, sanitize_doc_id.
All fixtures from conftest.py.
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.core.security import (
    FileTypeValidator, MalwareScanner, PIIRedactor,
    AuditLedger, SecurityGate, SecurityViolation,
    sanitize_path, sanitize_doc_id,
)

PDF_MAGIC  = b"%PDF-1.4 clean document content for testing purposes"
JPEG_MAGIC = b"\xff\xd8\xff header content"
EICAR      = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
JS_EVIL    = b"%PDF-1.4\n/JS javascript:alert(1)/JavaScript action"
EXE_BYTES  = b"\x4d\x5a\x90\x00embedded executable content here"


class TestFileTypeValidator:
    def setup_method(self):
        self.v = FileTypeValidator()

    def test_pdf_detected(self):
        assert self.v.detect_mime(PDF_MAGIC) == "application/pdf"

    def test_jpeg_detected(self):
        assert self.v.detect_mime(JPEG_MAGIC) == "image/jpeg"

    def test_unknown_type_returns_octet_stream(self):
        assert self.v.detect_mime(b"\x00\x01\x02\x03") == "application/octet-stream"

    def test_valid_pdf_validate_passes(self):
        mime = self.v.validate(PDF_MAGIC, "report.pdf", "doc_001")
        assert mime == "application/pdf"

    def test_unknown_type_validate_raises(self):
        with pytest.raises(SecurityViolation) as exc:
            self.v.validate(b"\x00\x01\x02\x03", "evil.exe", "doc_exe")
        assert exc.value.check == "file_type_validation"

    def test_exe_magic_raises(self):
        with pytest.raises(SecurityViolation):
            self.v.validate(EXE_BYTES, "trojan.pdf", "doc_exe")

    def test_filename_irrelevant_only_magic_matters(self):
        # A file named .exe but with PDF magic should pass file-type check
        mime = self.v.validate(PDF_MAGIC, "sneaky.exe", "doc_001")
        assert mime == "application/pdf"


class TestMalwareScanner:
    def setup_method(self):
        self.scanner = MalwareScanner()

    def test_eicar_detected(self):
        with pytest.raises(SecurityViolation) as exc:
            self.scanner.scan(EICAR, "test.pdf", "doc_eicar")
        assert exc.value.check == "eicar_signature"

    def test_clean_pdf_passes(self):
        self.scanner.scan(PDF_MAGIC, "clean.pdf", "doc_clean")  # no raise

    def test_javascript_heuristic_detected(self):
        with pytest.raises(SecurityViolation) as exc:
            self.scanner.scan(JS_EVIL, "evil.pdf", "doc_js")
        assert "heuristic" in exc.value.check

    def test_embedded_exe_detected(self):
        payload = PDF_MAGIC + b"\x00" * 100 + b"MZ\x90\x00embedded"
        with pytest.raises(SecurityViolation) as exc:
            self.scanner.scan(payload, "dropper.pdf", "doc_exe")
        assert "heuristic" in exc.value.check

    def test_powershell_pattern_detected(self):
        payload = PDF_MAGIC + b" powershell -enc abc123"
        with pytest.raises(SecurityViolation):
            self.scanner.scan(payload, "ps.pdf", "doc_ps")

    def test_security_violation_has_doc_id(self):
        with pytest.raises(SecurityViolation) as exc:
            self.scanner.scan(EICAR, "test.pdf", "my_doc_id")
        assert exc.value.doc_id == "my_doc_id"


class TestPIIRedactor:
    def setup_method(self):
        self.r = PIIRedactor()

    def test_email_redacted(self):
        text = "Contact john.doe@example.com for help."
        redacted, records = self.r.redact(text, "doc_1")
        assert "john.doe@example.com" not in redacted
        assert len(records) > 0

    def test_ssn_redacted(self):
        text = "SSN is 123-45-6789."
        redacted, records = self.r.redact(text, "doc_2")
        assert "123-45-6789" not in redacted

    def test_phone_redacted(self):
        text = "Call 555-867-5309 for support."
        redacted, records = self.r.redact(text, "doc_3")
        assert "555-867-5309" not in redacted

    def test_credit_card_redacted(self):
        text = "Card number 4111111111111111 was charged."
        redacted, records = self.r.redact(text, "doc_4")
        assert "4111111111111111" not in redacted

    def test_clean_text_unchanged(self):
        text = "The quarterly revenue was ETB 4.2 billion."
        redacted, records = self.r.redact(text, "doc_5")
        assert "4.2 billion" in redacted
        assert len(records) == 0

    def test_empty_text_returns_empty(self):
        redacted, records = self.r.redact("", "doc_6")
        assert redacted == ""
        assert records == []

    def test_records_have_entity_type(self):
        text = "Email: test@test.com"
        _, records = self.r.redact(text, "doc_7")
        assert all("entity_type" in r for r in records)


class TestAuditLedger:
    def test_append_and_read(self, tmp_path):
        ledger = AuditLedger(path=tmp_path / "audit.jsonl")
        ledger.append("TEST_EVENT", {"doc_id": "d1", "data": "value"})
        entries = ledger.read_all()
        assert len(entries) == 1
        assert entries[0]["event"] == "TEST_EVENT"
        assert entries[0]["payload"]["doc_id"] == "d1"

    def test_chain_valid_on_multiple_entries(self, tmp_path):
        ledger = AuditLedger(path=tmp_path / "audit.jsonl")
        for i in range(5):
            ledger.append(f"EVENT_{i}", {"seq": i})
        is_valid, msg = ledger.verify_chain()
        assert is_valid, f"Chain broken: {msg}"

    def test_empty_ledger_is_valid(self, tmp_path):
        ledger = AuditLedger(path=tmp_path / "empty.jsonl")
        is_valid, msg = ledger.verify_chain()
        assert is_valid

    def test_tamper_detected(self, tmp_path):
        ledger = AuditLedger(path=tmp_path / "audit.jsonl")
        ledger.append("ORIGINAL", {"data": "original"})
        ledger.append("SECOND",   {"data": "second"})
        lines = ledger.path.read_text().splitlines()
        entry = json.loads(lines[0])
        entry["payload"]["data"] = "TAMPERED"
        lines[0] = json.dumps(entry)
        ledger.path.write_text("\n".join(lines) + "\n")
        is_valid, msg = ledger.verify_chain()
        assert not is_valid

    def test_prev_hash_links_entries(self, tmp_path):
        ledger = AuditLedger(path=tmp_path / "audit.jsonl")
        ledger.append("E1", {"x": 1})
        ledger.append("E2", {"x": 2})
        entries = ledger.read_all()
        # Second entry's prev_hash must not be genesis (since first entry exists)
        assert entries[1]["prev_hash"] != "0" * 64


class TestSecurityGate:
    def test_ingest_clean_pdf_returns_meta(self, tmp_path):
        gate = SecurityGate(ledger=AuditLedger(tmp_path / "a.jsonl"))
        meta = gate.ingest(PDF_MAGIC, "report.pdf", "doc_001")
        assert meta["doc_id"] == "doc_001"
        assert meta["mime_type"] == "application/pdf"
        assert "content_hash" in meta
        assert len(meta["content_hash"]) == 64

    def test_ingest_malware_raises(self, tmp_path):
        gate = SecurityGate(ledger=AuditLedger(tmp_path / "a.jsonl"))
        with pytest.raises(SecurityViolation):
            gate.ingest(EICAR, "eicar.pdf", "doc_evil")

    def test_ingest_generates_doc_id_if_missing(self, tmp_path):
        gate = SecurityGate(ledger=AuditLedger(tmp_path / "a.jsonl"))
        meta = gate.ingest(PDF_MAGIC, "report.pdf", "")
        assert meta["doc_id"]  # auto-generated

    def test_ingest_logs_to_ledger(self, tmp_path):
        ledger = AuditLedger(tmp_path / "a.jsonl")
        gate = SecurityGate(ledger=ledger)
        gate.ingest(PDF_MAGIC, "report.pdf", "doc_log_test")
        entries = ledger.read_all()
        assert any("SECURITY_GATE" in e["event"] for e in entries)

    def test_redact_text_logs_to_ledger(self, tmp_path):
        ledger = AuditLedger(tmp_path / "a.jsonl")
        gate = SecurityGate(ledger=ledger)
        gate.redact_text("Contact admin@bank.et for support.", "doc_pii")
        entries = ledger.read_all()
        assert any("PII_REDACTION" in e["event"] for e in entries)


class TestSanitizeHelpers:
    def test_path_traversal_blocked(self, tmp_path):
        with pytest.raises(SecurityViolation) as exc:
            sanitize_path("../../etc/passwd", tmp_path)
        assert exc.value.check == "path_traversal"

    def test_valid_subpath_allowed(self, tmp_path):
        result = sanitize_path("profiles/doc_001.json", tmp_path)
        assert str(result).startswith(str(tmp_path.resolve()))

    def test_absolute_outside_base_blocked(self, tmp_path):
        with pytest.raises(SecurityViolation):
            sanitize_path("/etc/shadow", tmp_path)

    def test_doc_id_alphanumeric_passes(self):
        assert sanitize_doc_id("valid_doc-001") == "valid_doc-001"

    def test_doc_id_strips_slashes(self):
        result = sanitize_doc_id("../../etc/passwd")
        assert "/" not in result
        assert "." not in result

    def test_doc_id_strips_special_chars(self):
        result = sanitize_doc_id("doc<script>alert(1)</script>")
        assert "<" not in result
        assert ">" not in result
