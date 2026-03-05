# The Document Intelligence Refinery (v2.0)

Engineering Agentic Pipelines for Unstructured Document Extraction at Enterprise Scale.

## Overview

This repository implements a **5-stage agentic pipeline** that ingests heterogeneous documents (PDF, DOCX, Images) and emits structured, queryable, spatially-indexed knowledge. It features a **Constraint Enforcement System (CES)** with confidence-gated escalation strategies.

## Architecture

| Stage | Agent             | Purpose                                             |
| ----- | ----------------- | --------------------------------------------------- |
| 1     | Triage Agent      | Document profiling (origin, layout, domain)         |
| 2     | Extraction Layer  | Multi-strategy (Fast/Layout/Vision) with escalation |
| 3     | Chunking Engine   | Semantic LDU generation with 5 constitutional rules |
| 4     | PageIndex Builder | Hierarchical navigation tree                        |
| 5     | Query Agent       | RAG + SQL + Provenance interface                    |

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package
pip install -e .

# Download spaCy model
python -m spacy download en_core_web_sm
```

# update comming
