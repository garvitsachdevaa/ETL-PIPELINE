
# ETL Pipeline

A modular, production-ready **Extract → Transform → Load** pipeline for processing diverse document formats. It combines smart MIME-type routing, VLM-powered layout detection, block-aware OCR, multi-strategy chunking, and an interactive Streamlit UI into a single cohesive system.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Supported Formats](#supported-formats)
6. [Installation](#installation)
7. [Usage](#usage)
   - [Web Interface](#web-interface)
   - [Command Line](#command-line)
   - [Programmatic API](#programmatic-api)
8. [Chunking Strategies](#chunking-strategies)
9. [Output Structure](#output-structure)
10. [Configuration](#configuration)
11. [Development Guide](#development-guide)
12. [Dependencies](#dependencies)
13. [License](#license)

---

## Overview

The ETL Pipeline ingests raw files of virtually any format and produces clean, structured, chunked output ready for downstream tasks such as vector indexing, knowledge-graph construction, or LLM fine-tuning.

**Processing flow:**

```
File Upload
    │
    ▼
Ingestion & MIME Detection (loader + detector)
    │
    ├─► Text / Mixed  ──► Text/Mixed Handler ──► Structured Sections
    │
    └─► Binary (PDF/Image) ──► VLM Layout Detection (PaliGemma 2)
                                    │
                                 Block-Aware OCR (Chandra)
                                    │
                                 OCR Spell Correction
                                    │
    ┌──────────────────────────────────────────────┘
    │
    ▼
Chunking Engine (Line / Paragraph / Section / BERTopic Context)
    │
    ▼
SLM Entity & Relation Extraction (Qwen2.5 / Phi-3.5 / Mistral-7B)
    │
    ▼
Fuzzy Deduplication (RapidFuzz)
    │
    ▼
Structured JSON Output  ──►  Streamlit UI / Batch Export
```

---

## Key Features

| Category | Details |
|---|---|
| **Format Detection** | Automatic MIME type detection; pure vs. mixed content classification |
| **Text Parsing** | Plain text, CSV (delimiter-aware), JSON, HTML, XML, Markdown |
| **Binary Parsing** | DOCX (headings / lists / tables), XLSX (multi-worksheet), PDF, Images |
| **VLM Layout** | PaliGemma 2 (3B / 10B) detects semantic block boundaries on each page |
| **OCR** | Chandra OCR with per-block cropping; PyPDF2 / PyMuPDF fallback |
| **Spell Correction** | SymSpell + custom OCR noise patterns + optional BERT correction |
| **Chunking** | Line, Paragraph, Section, or BERTopic semantic context chunking |
| **Entity Extraction** | Fine-tuned SLM extracts entities + typed relations per chunk |
| **Deduplication** | RapidFuzz fuzzy merging with canonical entity resolution |
| **UI** | Streamlit web app — single-file, batch, and free-text modes |
| **Batch Processing** | Parallel batch processor with JSON output per batch |
| **Metadata** | Confidence scores, source format, extraction method, processing time |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          ETL PIPELINE — PHASE 2                         │
└──────────────────────────────────────────────────────────────────────────┘

 UPLOAD  (PDF / Image / DOCX / XLSX / Text / CSV / JSON / HTML / XML)
          │
          ▼
 ┌─────────────────────┐
 │  Ingestion Layer    │  loader.py → detector.py → validator.py → router.py
 └────────┬────────────┘
          │
    ┌─────┴──────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
 Text / Mixed Handler                     Binary Handler
 (text_handler / mixed_handler)           (binary_handler)
    │                                             │
    │  Parsers:                           Step 1: VLM Layout Detection
    │  • plain.py  • csv.py              (paligemma_adapter.py)
    │  • json.py   • html.py             → List[LayoutBlock] with bboxes
    │  • markdown.py                             │
    │                                    Step 2: Block-Aware Chandra OCR
    │                                    (chandra_adapter.py)
    │                                    → List[OcrBlock]
    │                                            │
    │                                    Step 3: Spell Correction
    │                                    (SymSpell + BERT)
    │                                            │
    └──────────────┬──────────────────────────────┘
                   │  Unified Document Object
                   ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                    Post-Processing Pipeline                 │
 │                                                             │
 │  Step 4: Chunking Engine (chunking/strategies.py)           │
 │    Line | Paragraph | Section | BERTopic Context            │
 │                      │                                       │
 │  Step 5: SLM Entity & Relation Extraction                   │
 │    Entities = {id, text, type, span}                        │
 │    Relations = {head, tail, type}                           │
 │                      │                                       │
 │  Step 6: Fuzzy Deduplication (RapidFuzz, threshold ≥ 88%)  │
 │    Canonical entity resolution across all chunks            │
 └──────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
              ExtractionResult JSON
              Streamlit UI  /  Batch Export
```

---

## Project Structure

```
ETL-PIPELINE/
├── app.py                          # Top-level app entry point
├── requirements.txt                # Root-level requirements
├── planner.md                      # Phase 2 implementation planner
└── etl_pipeline/
    ├── main.py                     # CLI entry point
    ├── requirements.txt            # Pipeline-specific dependencies
    ├── conftest.py                 # Pytest configuration
    │
    ├── common/
    │   ├── errors.py               # Custom exception classes
    │   └── utils.py                # Shared utility functions
    │
    ├── ingestion/
    │   ├── loader.py               # File loading & byte-level reading
    │   ├── detector.py             # MIME type & format detection
    │   ├── validator.py            # Content integrity validation
    │   ├── router.py               # Routes documents to correct handler
    │   ├── batch_processor.py      # Parallel batch processing
    │   ├── metadata.py             # Metadata extraction helpers
    │   └── schemas.py              # Pydantic schemas for ingestion
    │
    ├── handlers/
    │   ├── text_handler.py         # Plain text / structured text pipeline
    │   ├── binary_handler.py       # Binary document routing (PDF/image)
    │   ├── mixed_handler.py        # Mixed content (text + binary regions)
    │   ├── docx_handler.py         # DOCX: headings, lists, tables
    │   ├── xlsx_handler.py         # XLSX: multi-sheet, formulas, types
    │   ├── marker_handler.py       # Marker-based document segmentation
    │   ├── text_schema.py          # TextDocument schema
    │   ├── binary_schema.py        # BinaryDocument / Page / Region schema
    │   │
    │   ├── parsers/                # Text format parsers
    │   │   ├── plain.py            # Plain text parser
    │   │   ├── csv.py              # CSV with delimiter detection
    │   │   ├── json.py             # JSON with validation
    │   │   ├── html.py             # HTML (BeautifulSoup)
    │   │   ├── markdown.py         # Markdown parser
    │   │   └── dispatcher.py       # Routes to correct parser by MIME
    │   │
    │   ├── ocr/                    # OCR integration
    │   │   ├── chandra_adapter.py  # Chandra OCR CLI adapter
    │   │   ├── dispatcher.py       # Routes images/PDFs to OCR
    │   │   ├── image.py            # Image pre-processing helpers
    │   │   └── pdf.py              # PDF rasterisation for OCR
    │   │
    │   └── vlm/                    # Vision-Language Model layout detection
    │       ├── layout_detector.py      # Orchestrates VLM detection
    │       ├── paligemma_adapter.py    # PaliGemma 2 adapter
    │       ├── doclayout_yolo_adapter.py # DocLayout-YOLO adapter
    │       └── block_schema.py         # LayoutBlock / OcrBlock schemas
    │
    ├── chunking/
    │   ├── chunker.py              # Chunking engine façade
    │   ├── strategies.py           # Four strategies (line/para/section/context)
    │   └── schemas.py              # Chunk / ContextGroup schemas
    │
    ├── outputs/
    │   └── batch/                  # Saved batch output JSON files
    │
    ├── tests/
    │   ├── __init__.py
    │   └── test_ocr_routing.py     # OCR routing tests
    │
    └── ui/
        └── streamlit_app_unified.py  # Unified Streamlit web interface
```

---

## Supported Formats

### Text & Structured Documents
| Format | Extension(s) | Notes |
|---|---|---|
| Plain Text | `.txt` | Auto encoding detection (chardet) |
| CSV | `.csv` | Delimiter auto-detection |
| JSON | `.json` | Schema validation |
| HTML | `.html`, `.htm` | Tag-aware extraction via BeautifulSoup |
| XML | `.xml` | Namespace-aware |
| Markdown | `.md` | Heading hierarchy preserved |

### Binary Documents
| Format | Extension(s) | Notes |
|---|---|---|
| Word | `.docx` | Headings, lists, tables, styles |
| Excel | `.xlsx` | Multi-worksheet, data types, formulas |
| PDF | `.pdf` | VLM layout + Chandra OCR; PyPDF2 fallback |
| Images | `.jpg`, `.png`, `.tiff`, `.bmp`, `.gif` | Chandra OCR with pre-processing |

### Mixed Content
- HTML pages with embedded images
- XML documents mixing structured data and prose
- Documents containing both text regions and image regions

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU with ≥ 8 GB VRAM (required for VLM / SLM features)
- Chandra OCR CLI installed and on `PATH`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/garvitsachdevaa/ETL-PIPELINE.git
cd ETL-PIPELINE

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
cd etl_pipeline
pip install -r requirements.txt
```

> **Note**: `torch`, `transformers`, `accelerate`, and `bitsandbytes` are required for VLM layout detection. Make sure your CUDA toolkit matches your PyTorch build.

---

## Usage

### Web Interface

```bash
cd etl_pipeline
streamlit run ui/streamlit_app_unified.py
```

The Streamlit app offers three modes:
- **Single File** — upload one document, choose chunking strategy, view results
- **Batch** — upload a directory of documents for parallel processing
- **Free Text** — paste raw text and apply chunking / entity extraction inline

### Command Line

```bash
cd etl_pipeline

# Process a single file and print JSON output
python main.py --file path/to/document.pdf

# Batch-process a directory
python main.py --batch input_dir/ output_dir/

# Enable verbose logging
python main.py --batch input/ output/ --verbose

# Show help
python main.py --help
```

### Programmatic API

```python
import sys
sys.path.insert(0, "etl_pipeline")

from ingestion.loader import DocumentLoader
from ingestion.router import DocumentRouter
from chunking.chunker import Chunker

# 1. Load the document
loader = DocumentLoader()
doc = loader.load_file("report.pdf")

# 2. Route to the appropriate handler
router = DocumentRouter()
processed = router.route_document(doc)

# 3. Chunk the output
chunker = Chunker(strategy="paragraph")   # line | paragraph | section | context
chunks = chunker.chunk(processed)

for chunk in chunks:
    print(f"[{chunk.chunk_id}] {chunk.text[:120]}")
    print(f"  strategy : {chunk.strategy}")
    print(f"  metadata : {chunk.metadata}")
```

---

## Chunking Strategies

The chunking engine (`chunking/strategies.py`) implements four strategies, selectable per document or per batch:

| Strategy | Granularity | Description |
|---|---|---|
| **Line** (`line`) | Finest | Splits on newlines, then sentence boundaries (`.`, `!`, `?`) |
| **Paragraph** (`paragraph`) | Medium | Splits on blank lines (`\n\n`), preserving paragraph units |
| **Section** (`section`) | Structural | Detects headings via Markdown `##`, numbered lists, `ALL CAPS` titles, and `Chapter / Section` keywords |
| **Context** (`context`) | Semantic | Embeds paragraphs with `sentence-transformers`, reduces with UMAP, clusters with HDBSCAN via BERTopic; paragraphs in the same topic cluster form one chunk |

> The **context** strategy requires a GPU and a minimum of 4 paragraphs to activate BERTopic. For smaller inputs it falls back to paragraph splitting automatically.

---

## Output Structure

Every processed document returns a structured JSON object:

```json
{
  "document_id": "3f7a1c2d-...",
  "source_file": "annual_report.pdf",
  "processing_time_s": 4.71,
  "chunks": [
    {
      "chunk_id": "a1b2c3d4-...",
      "text": "Extracted and cleaned text content ...",
      "strategy": "section",
      "metadata": {
        "source_format": "pdf",
        "page": 2,
        "section_heading": "Financial Highlights",
        "extraction_method": "chandra_ocr",
        "confidence": 0.94
      },
      "entities": [
        { "id": "e1", "text": "Acme Corp", "type": "ORG", "span": [0, 9] }
      ],
      "relations": [
        { "head": "e1", "tail": "e2", "type": "FOUNDED_BY" }
      ]
    }
  ],
  "metadata": {
    "total_chunks": 18,
    "total_entities": 42,
    "total_relations": 11,
    "vlm_layout_used": true,
    "ocr_spell_corrected": true
  }
}
```

Batch outputs are saved to `etl_pipeline/outputs/batch/<batch_id>.json`.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CHANDRA_OCR_ENDPOINT` | `localhost:50051` | gRPC endpoint for the Chandra OCR service |
| `HF_TOKEN` | — | HuggingFace token for gated models (PaliGemma 2) |
| `LOG_LEVEL` | `INFO` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ETL_MAX_WORKERS` | `4` | Number of parallel workers for batch processing |
| `BERTOPIC_RANDOM_SEED` | `42` | Reproducibility seed for BERTopic context chunking |

### Model Settings

| Setting | Location | Description |
|---|---|---|
| PaliGemma model ID | `vlm/paligemma_adapter.py` | Switch between `3b` and `10b` variants |
| SLM model ID | `postprocessing/slm_extractor.py` | `Qwen2.5-7B`, `Phi-3.5`, or `Mistral-7B` |
| Fuzzy dedup threshold | `postprocessing/deduplicator.py` | Default `88` (token_set_ratio %) |
| Max context groups | `chunking/strategies.py` | Default `8` BERTopic groups per document |

---

## Development Guide

### Adding a New File Format

1. **Create a parser** in `handlers/parsers/` (see `csv.py` as a template).
2. **Register the MIME type** in `handlers/parsers/dispatcher.py`.
3. **Add MIME detection** in `ingestion/detector.py` if needed.
4. **Write tests** in `etl_pipeline/tests/`.

### Adding a New Chunking Strategy

1. Implement a function `chunk_by_<name>(segments) -> List[Chunk]` in `chunking/strategies.py`.
2. Add the strategy key to the dispatcher in `chunking/chunker.py`.
3. Expose it in the Streamlit UI dropdown in `ui/streamlit_app_unified.py`.

### Running Tests

```bash
cd etl_pipeline
pytest tests/ -v
```

### Code Style

The project uses standard Python conventions:
- Type hints throughout (`from __future__ import annotations`)
- Pydantic v2 schemas for all data models
- Module-level `logger = logging.getLogger(__name__)` for logging
- Docstrings on all public functions and classes

---

## Dependencies

### Core
| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` | Data processing |
| `chardet` | Encoding detection |
| `beautifulsoup4` / `lxml` | HTML & XML parsing |

### Document Processing
| Package | Purpose |
|---|---|
| `python-docx` | DOCX extraction |
| `openpyxl` | XLSX extraction |
| `PyPDF2` / `PyMuPDF` | PDF text & rasterisation |
| `Pillow` | Image pre-processing |
| `chandra-ocr` | GPU-accelerated OCR |

### Phase 2 — AI / ML
| Package | Purpose |
|---|---|
| `transformers` | PaliGemma 2 VLM inference |
| `accelerate` | Efficient GPU loading |
| `bitsandbytes` | 4-bit / 8-bit quantisation |
| `torch` | Deep learning backend |
| `sentence-transformers` | Paragraph embeddings for BERTopic |
| `bertopic` | Semantic context clustering |
| `rapidfuzz` | Fuzzy entity deduplication |
| `symspellpy` | Fast OCR spell correction |
| `huggingface_hub` | Model downloads & HF token auth |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes with clear messages.
4. Open a pull request describing what you changed and why.

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/garvitsachdevaa/ETL-PIPELINE/issues).
