"""
SecurityLayer — TRP1 Week 3: Document Intelligence Refinery

Provides:
  - Malware scanning (file-type + magic byte validation, optional ClamAV)
  - PII redaction via Microsoft Presidio (with spaCy + regex fallback)
  - Encryption at rest (Fernet symmetric encryption)
  - Immutable append-only audit ledger with SHA-256 chain linking
  - Input sanitization and path-traversal protection
  - SecurityMiddleware decorator wrapping entire pipeline stages
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from src.core.policy_engine import PolicyViolation

logger = logging.getLogger(__name__)

# Optional heavy dependencies — graceful degradation if not installed
try:
    from cryptography.fernet import Fernet
    _FERNET_AVAILABLE = True
except ImportError:
    _FERNET_AVAILABLE = False
    logger.warning("cryptography not installed — encryption at rest disabled")

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False
    logger.warning("presidio not installed — falling back to regex PII redaction")

try:
    import clamd
    _CLAMD_AVAILABLE = True
except ImportError:
    _CLAMD_AVAILABLE = False
    logger.info("clamd not installed — using magic-byte malware heuristics only")

# ─────────────────────────────────────────────────────────────────────────────
# Constants & allowed types
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain",
    "text/csv",
    "image/png",
    "image/jpeg",
    "image/tiff",
}

MAGIC_SIGNATURES: dict[bytes, str] = {
    b"%PDF":             "application/pdf",
    b"PK\x03\x04":      "application/zip",  # Office Open XML
    b"\xff\xd8\xff":    "image/jpeg",
    b"\x89PNG":         "image/png",
    b"II\x2a\x00":      "image/tiff",
    b"MM\x00\x2a":      "image/tiff",
}

MALWARE_SIGNATURES = [
    b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE",
    b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR",
]

SUSPICIOUS_PATTERNS = [
    re.compile(rb"javascript:", re.IGNORECASE),
    re.compile(rb"/JS\b"),
    re.compile(rb"/Launch\b"),
    re.compile(rb"cmd\.exe", re.IGNORECASE),
    re.compile(rb"powershell", re.IGNORECASE),
]

PII_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL",   re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("PHONE",   re.compile(r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("SSN",     re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT",  re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b")),
]


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class SecurityViolation(Exception):
    """Raised when a document fails security checks."""
    def __init__(self, check: str, detail: str, doc_id: str = "") -> None:
        self.check = check
        self.detail = detail
        self.doc_id = doc_id
        super().__init__(f"[SecurityViolation] check={check!r} doc={doc_id!r}: {detail}")


class PathTraversalError(SecurityViolation):
    def __init__(self, path: str) -> None:
        super().__init__("path_traversal", f"Dangerous path detected: {path!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Malware scanner
# ─────────────────────────────────────────────────────────────────────────────

class MalwareScanner:
    """
    Multi-layer malware detection:
      1. ClamAV daemon (if clamd is available and running)
      2. Known malware byte-signature matching
      3. Suspicious PDF pattern heuristics
    """

    def scan(self, file_bytes: bytes, filename: str = "", doc_id: str = "") -> None:
        """Raises SecurityViolation if malware is detected."""
        self._clamav_scan(file_bytes, filename, doc_id)
        self._signature_scan(file_bytes, doc_id)
        self._heuristic_scan(file_bytes, doc_id)
        logger.debug("Malware scan passed for %s", doc_id or filename)

    def _clamav_scan(self, file_bytes: bytes, filename: str, doc_id: str) -> None:
        if not _CLAMD_AVAILABLE:
            return
        try:
            cd = clamd.ClamdUnixSocket()
            result = cd.instream(file_bytes)
            status, detail = result.get("stream", ("OK", ""))
            if status == "FOUND":
                raise SecurityViolation("clamav", f"ClamAV detected: {detail}", doc_id)
        except (clamd.ConnectionError, OSError):
            logger.warning("ClamAV daemon not reachable — skipping ClamAV scan")

    def _signature_scan(self, file_bytes: bytes, doc_id: str) -> None:
        for sig in MALWARE_SIGNATURES:
            if sig in file_bytes:
                raise SecurityViolation("eicar_signature",
                    f"Known malware signature detected", doc_id)
        # Detect embedded EXE (MZ header) inside non-EXE files
        if file_bytes[:2] != b"MZ" and b"MZ" in file_bytes[4:]:
            raise SecurityViolation("heuristic",
                "Embedded executable (MZ) detected in file", doc_id)

    def _heuristic_scan(self, file_bytes: bytes, doc_id: str) -> None:
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern.search(file_bytes):
                raise SecurityViolation("heuristic",
                    f"Suspicious pattern found: {pattern.pattern[:40]!r}", doc_id)


# ─────────────────────────────────────────────────────────────────────────────
# File type validator
# ─────────────────────────────────────────────────────────────────────────────

class FileTypeValidator:
    """Validates file MIME type by magic bytes — not file extension."""

    def detect_mime(self, file_bytes: bytes) -> str:
        for magic, mime in MAGIC_SIGNATURES.items():
            if file_bytes[:len(magic)] == magic:
                # Disambiguate ZIP-based Office formats
                if mime == "application/zip":
                    if b"word/" in file_bytes[:8192]:
                        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    if b"xl/" in file_bytes[:8192]:
                        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    if b"ppt/" in file_bytes[:8192]:
                        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    return mime
                return mime
        return "application/octet-stream"

    def validate(self, file_bytes: bytes, filename: str = "", doc_id: str = "") -> str:
        mime = self.detect_mime(file_bytes)
        if mime not in ALLOWED_MIME_TYPES:
            raise SecurityViolation("file_type_validation",
                f"File type '{mime}' is not allowed. Filename: {filename!r}", doc_id)
        return mime


# ─────────────────────────────────────────────────────────────────────────────
# PII Redactor
# ─────────────────────────────────────────────────────────────────────────────

class PIIRedactor:
    """
    Redacts PII from text using Presidio if available, otherwise regex patterns.
    Configured entity list comes from RefineryPolicy.pii_entities.
    """

    def __init__(self, entities: list[str] | None = None) -> None:
        self.entities = entities or ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                                      "US_SSN", "CREDIT_CARD"]
        self._presidio_analyzer: Any = None
        self._presidio_anonymizer: Any = None
        if _PRESIDIO_AVAILABLE:
            try:
                self._presidio_analyzer = AnalyzerEngine()
                self._presidio_anonymizer = AnonymizerEngine()
            except Exception as exc:
                logger.warning("Presidio init failed (%s) — using regex fallback", exc)

    def redact(self, text: str, doc_id: str = "") -> tuple[str, list[dict]]:
        """Returns (redacted_text, list_of_redaction_records)."""
        if self._presidio_analyzer:
            return self._presidio_redact(text, doc_id)
        return self._regex_redact(text, doc_id)

    def _presidio_redact(self, text: str, doc_id: str) -> tuple[str, list[dict]]:
        from presidio_anonymizer.entities import OperatorConfig
        results = self._presidio_analyzer.analyze(
            text=text, entities=self.entities, language="en"
        )
        if not results:
            # Fall back to regex for what Presidio missed
            return self._regex_redact(text, doc_id)
        anonymized = self._presidio_anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})},
        )
        records = [{"entity_type": r.entity_type, "entity": r.entity_type,
                    "start": r.start, "end": r.end, "doc_id": doc_id}
                   for r in results]
        # Also apply regex SSN fallback in case Presidio missed it
        redacted_text = anonymized.text
        ssn_pattern = PII_REGEX_PATTERNS[2][1]  # SSN is index 2
        if ssn_pattern.search(redacted_text):
            for m in ssn_pattern.finditer(redacted_text):
                records.append({"entity_type": "US_SSN", "entity": "SSN",
                                 "start": m.start(), "end": m.end(), "doc_id": doc_id})
            redacted_text = ssn_pattern.sub("<REDACTED>", redacted_text)
        return redacted_text, records

    def _regex_redact(self, text: str, doc_id: str) -> tuple[str, list[dict]]:
        records: list[dict] = []
        for label, pattern in PII_REGEX_PATTERNS:
            for m in pattern.finditer(text):
                records.append({"entity_type": label, "entity": label,
                                 "start": m.start(), "end": m.end(), "doc_id": doc_id})
            text = pattern.sub("<REDACTED>", text)
        return text, records


# ─────────────────────────────────────────────────────────────────────────────
# Encryption
# ─────────────────────────────────────────────────────────────────────────────

class EncryptionManager:
    """Fernet symmetric encryption for .refinery/ artifacts."""

    _KEY_ENV_VAR = "REFINERY_FERNET_KEY"
    _KEY_FILE = Path(".refinery/.fernet.key")

    def __init__(self) -> None:
        if not _FERNET_AVAILABLE:
            logger.warning("cryptography not available — encryption disabled")
            self._fernet = None
            return
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        # 1. Environment variable (preferred in production / Docker)
        env_key = os.environ.get(self._KEY_ENV_VAR)
        if env_key:
            return env_key.encode()
        # 2. Key file (development)
        if self._KEY_FILE.exists():
            return self._KEY_FILE.read_bytes()
        # 3. Generate new key
        key = Fernet.generate_key()
        self._KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._KEY_FILE.write_bytes(key)
        self._KEY_FILE.chmod(0o600)
        logger.info("Generated new Fernet key at %s", self._KEY_FILE)
        return key

    def encrypt(self, data: bytes) -> bytes:
        if self._fernet is None:
            return data
        return self._fernet.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        if self._fernet is None:
            return token
        return self._fernet.decrypt(token)

    def encrypt_file(self, path: Path) -> Path:
        """Encrypt file in-place, append .enc suffix."""
        enc_path = path.with_suffix(path.suffix + ".enc")
        enc_path.write_bytes(self.encrypt(path.read_bytes()))
        path.unlink()
        return enc_path

    def decrypt_file(self, enc_path: Path) -> Path:
        """Decrypt .enc file, return original path."""
        original = Path(str(enc_path).removesuffix(".enc"))
        original.write_bytes(self.decrypt(enc_path.read_bytes()))
        return original


# ─────────────────────────────────────────────────────────────────────────────
# Immutable Audit Ledger
# ─────────────────────────────────────────────────────────────────────────────

class AuditLedger:
    """
    Append-only audit log with SHA-256 chain linking.
    Each entry contains the hash of the previous entry, making tampering detectable.
    """

    _LEDGER_PATH = Path(".refinery/audit_ledger.jsonl")

    def __init__(self, ledger_path: Path | None = None, path: Path | None = None) -> None:
        self._path = path or ledger_path or self._LEDGER_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = self._compute_chain_tip()

    @property
    def path(self) -> Path:
        return self._path

    def _compute_chain_tip(self) -> str:
        """SHA-256 of the entire current ledger file (genesis = 64 zeros)."""
        if not self._path.exists():
            return "0" * 64
        return hashlib.sha256(self._path.read_bytes()).hexdigest()

    def append(self, event_type: str, payload: dict[str, Any]) -> str:
        """Append an event. Returns the new entry hash."""
        entry = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "prev_hash": self._prev_hash,
            "payload": payload,
        }
        entry_bytes = json.dumps(entry, sort_keys=True).encode()
        entry_hash = hashlib.sha256(entry_bytes).hexdigest()
        entry["entry_hash"] = entry_hash

        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._prev_hash = entry_hash
        logger.debug("Audit ledger: %s event recorded (%s)", event_type, entry_hash[:12])
        return entry_hash

    def read_all(self) -> list[dict]:
        """Return all entries as a list of dicts."""
        if not self._path.exists():
            return []
        lines = self._path.read_text().splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        # Add 'event' as alias for 'event_type' for test compatibility
        for e in entries:
            if "event_type" in e and "event" not in e:
                e["event"] = e["event_type"]
        return entries

    def verify_chain(self) -> tuple[bool, str]:
        """Verify ledger integrity. Returns (True, 'ok') if chain is unbroken."""
        if not self._path.exists():
            return True, "ok"
        lines = self._path.read_text().splitlines()
        prev = "0" * 64
        for line in lines:
            entry = json.loads(line)
            if entry.get("prev_hash") != prev:
                msg = f"Chain broken at entry {entry.get('event_id')}"
                logger.error("Audit ledger %s", msg)
                return False, msg
            body = {k: v for k, v in entry.items() if k != "entry_hash"}
            expected = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            if entry.get("entry_hash") != expected:
                msg = f"Hash mismatch at {entry.get('event_id')}"
                logger.error("Audit ledger %s", msg)
                return False, msg
            prev = entry["entry_hash"]
        return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Input sanitization
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_path(raw_path: str, base_dir: Path) -> Path:
    """
    Prevent path traversal attacks. The resolved path must remain inside base_dir.
    Raises PathTraversalError if traversal is detected.
    """
    try:
        resolved = (base_dir / raw_path).resolve()
        base_resolved = base_dir.resolve()
        resolved.relative_to(base_resolved)  # raises ValueError if outside
        return resolved
    except (ValueError, OSError):
        raise PathTraversalError(raw_path)


def sanitize_doc_id(doc_id: str) -> str:
    """Strip non-alphanumeric chars except hyphens and underscores."""
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "", doc_id)
    if clean != doc_id:
        logger.warning("doc_id sanitized: %r → %r", doc_id, clean)
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# SecurityGate — unified entry point
# ─────────────────────────────────────────────────────────────────────────────

class SecurityGate:
    """
    Orchestrates all security checks at document ingestion.
    Must pass before any extraction begins.
    """

    def __init__(self, ledger: "AuditLedger | None" = None) -> None:
        self.scanner = MalwareScanner()
        self.validator = FileTypeValidator()
        self.redactor = PIIRedactor()
        self.encryptor = EncryptionManager()
        self.ledger = ledger or AuditLedger()

    def ingest(self, file_bytes: bytes, filename: str, doc_id: str = "") -> dict[str, Any]:
        """
        Full security gate. Returns metadata dict on success.
        Raises SecurityViolation or PolicyViolation on failure.
        """
        doc_id = sanitize_doc_id(doc_id or Path(filename).stem)
        t0 = time.time()

        # 0. File size check
        size_mb = len(file_bytes) / (1024 ** 2)
        max_mb = 10.0
        if size_mb > max_mb:
            raise PolicyViolation("max_file_size_mb",
                f"file is {size_mb:.1f} MB, limit is {max_mb:.1f} MB", doc_id)

        # 1. File type validation
        mime = self.validator.validate(file_bytes, filename, doc_id)

        # 2. Malware scan
        self.scanner.scan(file_bytes, filename, doc_id)

        # 3. Compute content hash
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        elapsed = time.time() - t0

        # 4. Audit log
        self.ledger.append("SECURITY_GATE_PASS", {
            "doc_id": doc_id,
            "filename": filename,
            "mime_type": mime,
            "file_size_bytes": len(file_bytes),
            "content_hash": content_hash,
            "scan_duration_s": round(elapsed, 4),
        })

        logger.info("SecurityGate PASSED for %s (%s, %d bytes) in %.2fs",
                    doc_id, mime, len(file_bytes), elapsed)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "mime_type": mime,
            "file_size_bytes": len(file_bytes),
            "content_hash": content_hash,
        }

    def redact_text(self, text: str, doc_id: str = "") -> tuple[str, list[dict]]:
        """Redact PII from extracted text. Returns (redacted_text, records)."""
        redacted, records = self.redactor.redact(text, doc_id)
        if records:
            self.ledger.append("PII_REDACTION", {
                "doc_id": doc_id,
                "redaction_count": len(records),
                "entities_found": list({r["entity"] for r in records}),
            })
        return redacted, records


# ─────────────────────────────────────────────────────────────────────────────
# SecurityMiddleware decorator
# ─────────────────────────────────────────────────────────────────────────────

_gate = SecurityGate()


def security_middleware(fn: Callable) -> Callable:
    """
    Wraps a pipeline stage function with audit logging and exception handling.
    Any SecurityViolation is logged and re-raised.

    Usage::

        @security_middleware
        def run_extraction(doc: ExtractedDocument) -> ExtractedDocument:
            ...
    """
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        stage_name = fn.__name__
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            _gate.ledger.append("STAGE_SUCCESS", {
                "stage": stage_name,
                "duration_s": round(elapsed, 4),
            })
            return result
        except SecurityViolation as exc:
            _gate.ledger.append("SECURITY_VIOLATION", {
                "stage": stage_name,
                "check": exc.check,
                "detail": exc.detail,
                "doc_id": exc.doc_id,
            })
            logger.error("SecurityViolation in %s: %s", stage_name, exc)
            raise
        except Exception as exc:
            elapsed = time.time() - t0
            _gate.ledger.append("STAGE_ERROR", {
                "stage": stage_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "duration_s": round(elapsed, 4),
            })
            raise
    return wrapper