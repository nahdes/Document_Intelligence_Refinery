"""
server.py — FastAPI backend for the Document Intelligence Refinery Dashboard.

Usage:
    uv run python server.py

Then open dashboard.html in your browser (or visit http://localhost:8000/dashboard).

Endpoints:
    POST /pipeline          — Upload PDF, run full pipeline, return results
    POST /query             — Ask a question about an already-processed document
    GET  /audit/{doc_id}    — Get full audit trail for a document
    GET  /dashboard         — Serve the dashboard HTML
    GET  /health            — Health check
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("refinery.server")

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise SystemExit(
        "\n❌  FastAPI/uvicorn not installed.\n"
        "Run:  uv add fastapi uvicorn python-multipart\n"
    )

# ── Profile converter: profile.py → schemas.py DocumentProfile ───────────────
def _convert_profile(p) -> Any:
    """
    Convert a src.models.profile.DocumentProfile (returned by TriageAgent)
    into a src.models.schemas.DocumentProfile (required by ExtractedDocument).

    Both classes have the same name but different fields.  We map the shared
    fields and provide safe defaults for anything extra schemas expects.
    """
    from src.models.schemas import (
        DocumentProfile as SchemasProfile,
        ExtractionStrategy,
        OriginType,
        LayoutComplexity,
        DomainHint,
    )

    # Map strategy letter A/B/C → ExtractionStrategy enum
    _smap = {
        "A": ExtractionStrategy.FAST,
        "B": ExtractionStrategy.LAYOUT,
        "C": ExtractionStrategy.VISION,
    }
    raw_strat = str(getattr(p, "recommended_strategy", "A")).upper()
    strategy  = _smap.get(raw_strat, ExtractionStrategy.FAST)

    # Safe enum coercions (profile.py stores string values via use_enum_values)
    def _ot(v):
        try: return OriginType(str(v))
        except: return OriginType.NATIVE_DIGITAL

    def _lc(v):
        try: return LayoutComplexity(str(v))
        except: return LayoutComplexity.SINGLE_COLUMN

    def _dh(v):
        try: return DomainHint(str(v))
        except: return DomainHint.GENERAL

    page_count   = int(getattr(p, "page_count", 1))
    cost_per_page = float(getattr(p, "estimated_cost_per_page", getattr(p, "total_estimated_cost", 0.0)) or 0.0)
    total_cost   = float(getattr(p, "total_estimated_cost", cost_per_page * page_count) or 0.0)

    return SchemasProfile(
        doc_id               = str(p.doc_id),
        filename             = str(p.filename),
        file_size_bytes      = int(getattr(p, "file_size_bytes", 0)),
        page_count           = page_count,
        mime_type            = str(getattr(p, "mime_type", "application/pdf")),
        origin_type          = _ot(p.origin_type),
        layout_complexity    = _lc(p.layout_complexity),
        domain_hint          = _dh(getattr(p, "domain_hint", "general")),
        language             = str(getattr(p, "language", "en")),
        language_confidence  = float(getattr(p, "language_confidence", 1.0)),
        recommended_strategy = strategy,
        estimated_cost_usd   = total_cost,
        avg_chars_per_page   = float(getattr(p, "avg_chars_per_page", 0.0)),
        avg_image_area_ratio = float(getattr(p, "image_area_ratio", 0.0)),
        has_tables           = bool(getattr(p, "has_tables", False)),
        has_figures          = bool(getattr(p, "has_figures", False)),
        content_hash         = str(getattr(p, "content_hash", "")),
    )


# ── In-memory session store (keyed by doc_id) ─────────────────────────────
_sessions: dict[str, dict[str, Any]] = {}

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Document Intelligence Refinery",
    version="1.0.0",
    description="Multi-strategy PDF pipeline with security & policy enforcement",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _clear_stale_ledgers():
    """Wipe any leftover dashboard ledgers from previous server runs."""
    import shutil
    ledger_dir = Path(".refinery/dashboard")
    if ledger_dir.exists():
        shutil.rmtree(ledger_dir, ignore_errors=True)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cleared stale dashboard ledgers")

# ── Dashboard static file ─────────────────────────────────────────────────
DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


@app.get("/dashboard", include_in_schema=False)
def serve_dashboard():
    if not DASHBOARD_PATH.exists():
        raise HTTPException(404, "dashboard.html not found next to server.py")
    return FileResponse(DASHBOARD_PATH, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(_sessions)}


# ── POST /pipeline ────────────────────────────────────────────────────────
@app.post("/pipeline")
async def run_pipeline(file: UploadFile = File(...)):
    """
    Upload a PDF and run the full pipeline.
    Returns profile, chunks, index, and audit trail.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return await _run_pipeline(tmp_path, file.filename, content)
    finally:
        tmp_path.unlink(missing_ok=True)


async def _run_pipeline(pdf_path: Path, filename: str, raw_bytes: bytes) -> JSONResponse:
    from src.core.constraint_enforcement import ConstraintEnforcementSystem
    from src.core.security import AuditLedger, SecurityViolation
    from src.core.policy_engine import PolicyViolation
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import ChunkingEngine
    from src.agents.indexer import PageIndexBuilder
    from src.models.schemas import ExtractionStrategy

    # Strategy letter → ExtractionStrategy enum
    STRATEGY_MAP = {
        "A": ExtractionStrategy.FAST,
        "B": ExtractionStrategy.LAYOUT,
        "C": ExtractionStrategy.VISION,
        "fast": ExtractionStrategy.FAST,
        "layout": ExtractionStrategy.LAYOUT,
        "vision": ExtractionStrategy.VISION,
    }

    # Per-run temp ledger
    ledger_dir = Path(".refinery/dashboard")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
    ledger_path = ledger_dir / f"{file_hash}.jsonl"
    # Clear previous ledger for same file
    ledger_path.unlink(missing_ok=True)

    ledger = AuditLedger(path=ledger_path)
    ces = ConstraintEnforcementSystem(audit_ledger=ledger)

    errors = []

    # ── Stage 1: Triage ──────────────────────────────────────────────────
    logger.info("[%s] Stage 1: Triage", filename)
    try:
        triage_profile = TriageAgent(ces=ces).profile(pdf_path)
        doc_id = triage_profile.doc_id
        # Convert profile.py DocumentProfile → schemas.py DocumentProfile
        # (two separate classes; ExtractedDocument.source_profile requires schemas version)
        profile = _convert_profile(triage_profile)
    except SecurityViolation as exc:
        raise HTTPException(400, f"Security check failed: [{exc.check}] {exc.detail}")
    except PolicyViolation as exc:
        raise HTTPException(400, f"Policy check failed: [{exc.rule}] {exc.detail}")
    except Exception as exc:
        logger.error("Triage failed: %s", exc)
        raise HTTPException(500, f"Triage failed: {exc}")

    # ── Stage 2: Extraction ───────────────────────────────────────────────
    logger.info("[%s] Stage 2: Extraction (strategy=%s)", filename, profile.recommended_strategy)
    extracted = None
    try:
        # Normalise recommended_strategy to ExtractionStrategy enum
        raw_strat = profile.recommended_strategy
        if not isinstance(raw_strat, ExtractionStrategy):
            raw_strat = STRATEGY_MAP.get(str(raw_strat), ExtractionStrategy.FAST)
            profile.recommended_strategy = raw_strat

        extracted = ExtractionRouter(ces=ces).extract(pdf_path, profile)
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        errors.append(f"Extraction: {exc}")

    # ── Stage 3: Chunking ─────────────────────────────────────────────────
    ldus = []
    if extracted:
        logger.info("[%s] Stage 3: Chunking", filename)
        try:
            ldus = ChunkingEngine(ces=ces).chunk(extracted)
        except Exception as exc:
            logger.error("Chunking failed: %s", exc)
            errors.append(f"Chunking: {exc}")

    # ── Stage 4: Page Index ───────────────────────────────────────────────
    index = None
    if ldus:
        logger.info("[%s] Stage 4: Indexing", filename)
        try:
            index = PageIndexBuilder(ces=ces).build(extracted, ldus)
        except Exception as exc:
            logger.error("Indexing failed: %s", exc)
            errors.append(f"Indexing: {exc}")

    # ── Serialize results ─────────────────────────────────────────────────
    profile_dict = _serialize_profile(triage_profile, profile)
    chunks_list = _serialize_chunks(ldus)
    index_dict = _serialize_index(index)
    audit_list = ledger.read_all()
    chain_valid, chain_msg = ledger.verify_chain()

    # Store session for /query endpoint
    _sessions[profile.doc_id] = {
        "ldus": ldus,
        "index": index,
        "ces": ces,
        "profile": profile,
    }

    logger.info("[%s] Pipeline complete — doc_id=%s, chunks=%d", filename, profile.doc_id, len(ldus))

    return JSONResponse({
        "doc_id": profile.doc_id,
        "profile": profile_dict,
        "chunks": chunks_list,
        "index": index_dict,
        "audit": audit_list,
        "chain_valid": chain_valid,
        "chain_msg": chain_msg,
        "errors": errors,
        "stats": {
            "page_count": profile.page_count,
            "chunk_count": len(ldus),
            "section_count": len(index.sections) if index else 0,
            "audit_entries": len(audit_list),
        },
    })


# ── POST /query ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    doc_id: str
    question: str


@app.post("/query")
def run_query(req: QueryRequest):
    """Ask a question about an already-processed document."""
    session = _sessions.get(req.doc_id)
    if not session:
        raise HTTPException(404, f"No session found for doc_id={req.doc_id!r}. Run /pipeline first.")

    from src.agents.query_agent import QueryAgent

    ldus = session["ldus"]
    index = session["index"]
    ces = session["ces"]

    if not ldus or not index:
        raise HTTPException(422, "Document has no chunks or index — extraction may have failed.")

    try:
        agent = QueryAgent(ldus=ldus, page_index=index, ces=ces)
        result = agent.query(req.question)
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(500, f"Query failed: {exc}")

    audit_list = ces.audit.read_all()
    chain_valid, chain_msg = ces.audit.verify_chain()

    return {
        "question": req.question,
        "answer": result.get("answer", str(result)),
        "sources": _serialize_sources(result.get("sources", [])),
        "confidence": result.get("confidence", 0.0),
        "audit": audit_list,
        "chain_valid": chain_valid,
    }


# ── GET /audit/{doc_id} ───────────────────────────────────────────────────
@app.get("/audit/{doc_id}")
def get_audit(doc_id: str):
    session = _sessions.get(doc_id)
    if not session:
        raise HTTPException(404, f"No session for doc_id={doc_id!r}")
    audit = session["ces"].audit.read_all()
    chain_valid, chain_msg = session["ces"].audit.verify_chain()
    return {"audit": audit, "chain_valid": chain_valid, "chain_msg": chain_msg}


# ── Serializers ───────────────────────────────────────────────────────────
def _serialize_profile(triage_p, schemas_p=None) -> dict:
    """Serialize for dashboard. triage_p = profile.py instance (has domain_hint etc.),
    schemas_p = schemas.py instance (has cost_estimate_usd, has_tables etc.)."""
    p = triage_p  # primary source for display fields
    s = schemas_p or triage_p  # fallback to same if not provided

    # cost: profile.py uses total_estimated_cost; schemas.py uses estimated_cost_usd
    cost = (
        float(getattr(s, "estimated_cost_usd", 0.0)) or
        float(getattr(p, "total_estimated_cost", 0.0)) or
        float(getattr(p, "estimated_cost_per_page", 0.0)) * int(getattr(p, "page_count", 1))
    )

    # strategy: show as A/B/C string for dashboard
    strat = str(getattr(p, "recommended_strategy", getattr(s, "recommended_strategy", "")))
    # if it came back as enum value (fast/layout/vision), map back to letter
    _rev = {"fast": "A", "layout": "B", "vision": "C"}
    strat_display = _rev.get(strat.lower(), strat)

    return {
        "doc_id":               str(p.doc_id),
        "filename":             str(getattr(p, "filename", "")),
        "page_count":           int(getattr(p, "page_count", 0)),
        "domain_hint":          str(getattr(p, "domain_hint", getattr(s, "domain_hint", ""))),
        "origin_type":          str(getattr(p, "origin_type", "")),
        "recommended_strategy": strat_display,
        "language":             str(getattr(p, "language", "en")),
        "language_confidence":  float(getattr(p, "language_confidence", 0.0)),
        "layout_complexity":    str(getattr(p, "layout_complexity", "")),
        "has_tables":           bool(getattr(s, "has_tables", getattr(p, "has_tables", False))),
        "has_figures":          bool(getattr(s, "has_figures", getattr(p, "has_figures", False))),
        "cost_estimate_usd":    cost,
    }


def _serialize_chunks(ldus) -> list[dict]:
    result = []
    for ldu in ldus:
        try:
            chunk_type = str(getattr(ldu, "chunk_type", "text"))
            result.append({
                "chunk_id":      str(getattr(ldu, "chunk_id", getattr(ldu, "ldu_id", ""))),
                "type":          chunk_type,
                "content":       str(getattr(ldu, "content", ""))[:500],
                "section":       str(getattr(ldu, "parent_section", "") or ""),
                "page":          int((getattr(ldu, "page_refs", [1]) or [1])[0]),
                "tokens":        int(getattr(ldu, "token_count", 0)),
                "content_hash":  str(getattr(ldu, "content_hash", ""))[:16],
                "pii_redacted":  bool(getattr(ldu, "pii_redacted", False)),
            })
        except Exception as exc:
            logger.warning("Could not serialize LDU: %s", exc)
    return result


def _serialize_index(index) -> dict | None:
    if not index:
        return None
    sections = []
    for s in getattr(index, "sections", []) or getattr(index, "root_sections", []) or []:
        sections.append({
            "title":      str(getattr(s, "title", "")),
            "page_start": int(getattr(s, "page_start", 1)),
            "page_end":   int(getattr(s, "page_end", 1)),
            "ldu_count":  len(getattr(s, "ldu_ids", []) or []),
            "summary":    str(getattr(s, "summary", ""))[:120],
            "entities":   list(getattr(s, "key_entities", []) or [])[:6],
        })
    return {
        "doc_id":   str(getattr(index, "doc_id", "")),
        "filename": str(getattr(index, "filename", "")),
        "sections": sections,
    }


def _serialize_sources(sources) -> list[dict]:
    result = []
    for s in sources or []:
        if isinstance(s, dict):
            result.append(s)
        else:
            result.append({
                "chunk_id":    str(getattr(s, "chunk_id", "")),
                "page_number": int(getattr(s, "page_number", 0)),
                "excerpt":     str(getattr(s, "excerpt", ""))[:200],
                "doc_id":      str(getattr(s, "doc_id", "")),
            })
    return result


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔬 Document Intelligence Refinery — Dashboard Server")
    print("────────────────────────────────────────────────────")
    print("  Dashboard → http://localhost:8000/dashboard")
    print("  API docs  → http://localhost:8000/docs")
    print("  Health    → http://localhost:8000/health")
    print("────────────────────────────────────────────────────\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)