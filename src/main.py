"""
Entry point — Document Intelligence Refinery CLI.
Usage:
  python -m src.main process path/to/document.pdf
  python -m src.main query --doc-id <id> "your question here"
  python -m src.main audit --doc-id <id> "The revenue was $4.2B"
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("refinery")


def process(file_path: str) -> None:
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import ChunkingEngine
    from src.agents.indexer import PageIndexBuilder

    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    logger.info("=== STAGE 1: Triage ===")
    triage = TriageAgent()
    profile = triage.profile(path)
    print(f"\nDocumentProfile:")
    print(json.dumps(profile.model_dump(mode="json"), indent=2, default=str))

    logger.info("=== STAGE 2: Extraction (strategy=%s) ===", profile.recommended_strategy.value)
    router = ExtractionRouter()
    doc = router.extract(path, profile)
    print(f"\nExtracted: {len(doc.text_blocks)} text blocks, {len(doc.tables)} tables")
    print(f"Confidence: {doc.confidence_score:.3f} | Cost: ${doc.cost_estimate_usd:.4f}")

    logger.info("=== STAGE 3: Chunking ===")
    chunker = ChunkingEngine()
    ldus = chunker.chunk(doc)
    print(f"\nChunked: {len(ldus)} LDUs")

    logger.info("=== STAGE 4: PageIndex ===")
    indexer = PageIndexBuilder()
    index = indexer.build(profile.doc_id, profile.filename, ldus, profile.page_count)
    print(f"\nPageIndex: {len(index.root_sections)} sections")
    for s in index.root_sections[:5]:
        print(f"  [{s.page_start}-{s.page_end}] {s.title[:60]}")

    print(f"\n✅ Pipeline complete. doc_id={profile.doc_id}")
    print(f"Profile saved to: .refinery/profiles/{profile.doc_id}.json")
    print(f"PageIndex saved to: .refinery/pageindex/{profile.doc_id}.json")


def query(doc_id: str, question: str) -> None:
    import json
    from pathlib import Path
    from src.agents.query_agent import QueryAgent
    from src.models.schemas import LDU, PageIndex

    index_path = Path(f".refinery/pageindex/{doc_id}.json")
    if not index_path.exists():
        logger.error("PageIndex not found for doc_id=%s. Run 'process' first.", doc_id)
        sys.exit(1)

    index = PageIndex.model_validate_json(index_path.read_text())
    agent = QueryAgent(ldus=[], page_index=index)
    result = agent.query(question)
    print(result.format_citation())


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Document Intelligence Refinery CLI")
    subparsers = parser.add_subparsers(dest="command")

    proc = subparsers.add_parser("process", help="Process a document through the full pipeline")
    proc.add_argument("file", help="Path to document file")

    qry = subparsers.add_parser("query", help="Query a processed document")
    qry.add_argument("--doc-id", required=True)
    qry.add_argument("question")

    args = parser.parse_args()

    if args.command == "process":
        process(args.file)
    elif args.command == "query":
        query(args.doc_id, args.question)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
