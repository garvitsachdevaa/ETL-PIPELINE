# Project Context — ETL Pipeline

> **Purpose of this file:** Hand this document to any LLM to give it complete, working context of this codebase — what it is, how every part fits together, what is already built, what is still planned, and all key design decisions.

---

## 1. What This Project Is

A **modular, GPU-accelerated ETL (Extract → Transform → Load) pipeline** built in Python. It ingests raw files of any common format, extracts clean structured text, chunks it intelligently, and runs Named Entity Recognition (NER) + Relation Extraction using a Small Language Model (SLM). The final output is a structured JSON object ready for vector databases, knowledge graphs, or LLM fine-tuning.

The system has a **Streamlit web UI** for interactive use and a **CLI + programmatic API** for automation and batch processing.

**GitHub:** `https://github.com/garvitsachdevaa/ETL-PIPELINE`  
**Active branch:** `main`  
**Last planner update:** February 19, 2026  
**GPU target:** 24 GB VRAM (local or HuggingFace Inference Endpoint)

---

## 2. Repository Layout

```
ETL-PIPELINE/
├── README.md                           ← Public-facing documentation
├── context.md                          ← THIS FILE (LLM context document)
├── planner.md                          ← Detailed Phase 2 implementation planner (954 lines)
├── app.py                              ← Top-level entry point
├── requirements.txt                    ← Root requirements
└── etl_pipeline/                       ← All source code lives here
    ├── main.py                         ← CLI entry point
    ├── conftest.py                     ← Pytest config
    ├── requirements.txt                ← Pipeline-specific deps
    ├── common/
    │   ├── errors.py                   ← Custom exceptions
    │   └── utils.py                    ← Shared helpers
    ├── ingestion/
    │   ├── loader.py                   ← File loading (bytes / path / raw text)
    │   ├── detector.py                 ← MIME + format detection (ext-first, then chardet)
    │   ├── validator.py                ← Content integrity checks
    │   ├── router.py                   ← Maps detected format → handler name string
    │   ├── batch_processor.py          ← Parallel batch processing
    │   ├── metadata.py                 ← Metadata extraction helpers
    │   └── schemas.py                  ← Pydantic schemas for ingestion objects
    ├── handlers/
    │   ├── text_handler.py             ← Text / structured text pipeline
    │   ├── binary_handler.py           ← Binary doc routing (PDF / image); calls VLM + OCR
    │   ├── mixed_handler.py            ← Mixed content (inline text + binary regions)
    │   ├── docx_handler.py             ← python-docx: headings, lists, tables, styles
    │   ├── xlsx_handler.py             ← openpyxl: multi-sheet, data types, formulas
    │   ├── marker_handler.py           ← Marker-based document segmentation
    │   ├── text_schema.py              ← TextDocument dataclass
    │   ├── binary_schema.py            ← BinaryDocument / Page / Region / Block schemas
    │   ├── parsers/
    │   │   ├── dispatcher.py           ← Routes to correct parser by MIME type
    │   │   ├── plain.py                ← Plain text
    │   │   ├── csv.py                  ← CSV with delimiter detection
    │   │   ├── json.py                 ← JSON with validation
    │   │   ├── html.py                 ← HTML via BeautifulSoup
    │   │   └── markdown.py             ← Markdown parser
    │   ├── ocr/
    │   │   ├── dispatcher.py           ← Routes images/PDFs to OCR, receives LayoutBlocks
    │   │   ├── chandra_adapter.py      ← Chandra OCR CLI adapter (block-aware in Phase 2)
    │   │   ├── image.py                ← Image pre-processing helpers
    │   │   └── pdf.py                  ← PDF rasterisation (PyMuPDF) for OCR
    │   └── vlm/                        ← NEW in Phase 2
    │       ├── __init__.py
    │       ├── paligemma_adapter.py    ← PaliGemma 2 model load + JSON-output inference
    │       ├── layout_detector.py      ← Orchestrates VLM over a full document
    │       ├── doclayout_yolo_adapter.py ← Alternative YOLO-based layout detector
    │       └── block_schema.py         ← LayoutBlock dataclass
    ├── chunking/
    │   ├── chunker.py                  ← Chunking engine façade (dispatches to strategies)
    │   ├── strategies.py               ← Four strategy functions (383 lines)
    │   └── schemas.py                  ← Chunk / ContextGroup / ChunkingResult dataclasses
    ├── outputs/
    │   └── batch/                      ← Saved batch output JSON files
    ├── tests/
    │   ├── __init__.py
    │   └── test_ocr_routing.py
    └── ui/
        └── streamlit_app_unified.py    ← Single unified Streamlit app
```

---

## 3. End-to-End Data Flow

```
File / Bytes / Raw Text
        │
        ▼
┌──────────────────────┐
│  ingestion/loader.py │  Reads file → produces (source_name, raw_bytes, raw_text)
└──────────┬───────────┘
           │
           ▼
┌────────────────────────┐
│  ingestion/detector.py │  detect_format(source_name, raw_bytes, raw_text)
│                        │  Priority: extension mapping → mimetypes → chardet → content sniff
│                        │  Returns: (format_type, mime_type, encoding)
│                        │  Format types: "text" | "structured" | "document" |
│                        │               "image" | "mixed" | "unknown"
└──────────┬─────────────┘
           │
           ▼
┌──────────────────────┐
│  ingestion/router.py │  Maps format_type → handler name string
│                      │  text/structured → "text_handler"
│                      │  image/document  → "binary_handler"
│                      │  mixed           → "mixed_handler"
│                      │  unknown         → triggers error in batch_processor
└──────────┬───────────┘
           │
     ┌─────┴────────────────────────────────────────────┐
     │                                                   │
     ▼                                                   ▼
Text/Mixed Handler                               Binary Handler
     │                                                   │
     │  Parsers dispatch by MIME:                 Phase 2 pipeline:
     │  plain.py / csv.py / json.py              ┌──────────────────────────┐
     │  html.py / markdown.py                    │ 1. VLM Layout Detection  │
     │                                           │    PaliGemma 2 → bbox    │
     │  → TextDocument                           │    per semantic block    │
     │    sections: List[Section]                ├──────────────────────────┤
     │    metadata: Dict                         │ 2. Block-Aware OCR       │
     │                                           │    Crop image per block  │
     │  XLSX → xlsx_handler.py → TextDocument    │    Chandra OCR per crop  │
     │  DOCX → docx_handler.py → TextDocument    ├──────────────────────────┤
     │                                           │ 3. OCR Spell Correction  │
     │                                           │    SymSpell + BERT       │
     │                                           └────────────┬─────────────┘
     │                                                        │
     │                                               BinaryDocument
     │                                               pages: List[Page]
     │                                               Page.blocks: List[Block]
     └───────────────────────┬────────────────────────────────┘
                             │ Unified document object
                             ▼
              ┌──────────────────────────────────┐
              │  Chunking Engine                 │
              │  chunking/chunker.py             │
              │                                  │
              │  Strategy A: Line (sentence)     │
              │  Strategy B: Paragraph/Section   │
              │  Strategy C: Page                │
              │  Strategy D: Context (BERTopic)  │
              │                                  │
              │  → ChunkingResult                │
              │    chunks: List[Chunk]           │
              │    context_groups (D only)       │
              └──────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  SLM Entity & Relation Extraction │
              │  postprocessing/slm_extractor.py  │
              │  Model: Qwen2.5-7B (planned)     │
              │                                  │
              │  → List[ChunkExtraction]          │
              │    entities: List[Entity]         │
              │    relations: List[Relation]      │
              └──────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  Fuzzy Deduplication              │
              │  postprocessing/deduplicator.py   │
              │  RapidFuzz token_set_ratio        │
              │  Union-Find canonical merge       │
              │                                  │
              │  → DedupedExtractionResult        │
              └──────────────────────────────────┘
                             │
                             ▼
              Streamlit UI / JSON Export / Batch file
```

---

## 4. Key Data Schemas

### ingestion layer
```python
# DocumentObject (from loader.py)
source_name: str
raw_bytes: bytes | None
raw_text: str | None
format_type: str        # text | structured | document | image | mixed | unknown
mime_type: str
encoding: str
```

### text handler output (text_schema.py)
```python
@dataclass
class Section:
    section_id: str
    content: str
    format_type: str      # heading | paragraph | table | list | code
    metadata: Dict[str, Any]

@dataclass
class TextDocument:
    document_id: str
    sections: List[Section]
    metadata: Dict[str, Any]   # source_file, processing_time, extraction_method, …
```

### binary handler output (binary_schema.py) — Phase 2 extended
```python
@dataclass
class Region:           # word-level OCR unit (EXISTING, unchanged)
    region_id: str
    text: str
    bbox: List[int]     # [x0, y0, x1, y1]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class Block:            # NEW in Phase 2
    block_id:       str
    title:          str         # Chandra-assigned heading
    label:          str         # VLM semantic label e.g. "user_profile"
    bbox:           List[int]
    regions:        List[Region]
    raw_text:       str         # direct OCR output
    corrected_text: str         # spell-corrected text
    confidence:     float
    metadata:       Dict[str, Any]

@dataclass
class Page:             # EXTENDED in Phase 2
    page_id:     str
    page_number: int
    blocks:      List[Block]    # replaces flat regions list
    metadata:    Dict[str, Any]
    # backwards-compat property:
    # regions = [r for b in self.blocks for r in b.regions]

@dataclass
class BinaryDocument:
    document_id: str
    pages: List[Page]
    metadata: Dict[str, Any]
```

### VLM output (block_schema.py)
```python
@dataclass
class LayoutBlock:
    block_id:    str          # uuid4
    label:       str          # e.g. "table", "figure", "header", "user_profile"
    bbox:        List[int]    # [x0, y0, x1, y1] in pixels
    confidence:  float
    page_number: int
```

### chunking output (chunking/schemas.py)
```python
@dataclass
class Chunk:
    text:              str
    method:            str      # line | paragraph | section | context
    chunk_index:       int
    chunk_id:          str      # uuid4
    metadata:          Dict[str, Any]  # source_block_id, source_page, strategy_tier, …
    similarity_score:  float    # cosine sim to BERTopic centroid (context mode only)
    related_chunk_ids: List[str]

@dataclass
class ContextGroup:             # context strategy only
    group_id:       str
    topic_id:       int
    topic_label:    str
    topic_words:    List[str]
    merged_chunk:   Chunk | None
    source_chunks:  List[Chunk]
    coherence_score: float      # intra-group cohesion 0–1

@dataclass
class ChunkingResult:
    method:          str
    chunks:          List[Chunk]
    context_groups:  List[ContextGroup]  # populated only for context strategy
    total_chunks:    int
    coherence_score: float               # overall silhouette score (context only)
    # property: slm_payload → Dict ready for SLM extractor
```

### SLM extraction output (postprocessing/extraction_schema.py — planned)
```python
@dataclass
class Entity:
    id:           str
    text:         str
    type:         str           # PERSON | ORG | PRODUCT | LOCATION | DATE | EVENT | ROLE | SKILL
    span:         List[int]     # [start_char, end_char] in chunk
    canonical_id: str | None    # set after deduplication

@dataclass
class Relation:
    head: str       # entity id
    tail: str       # entity id
    type: str       # WORKS_AT | FOUNDED_BY | LOCATED_IN | COLLABORATED_WITH |
                    # HAS_SKILL | PUBLISHED_ON | MENTIONS | PART_OF

@dataclass
class ChunkExtraction:
    chunk_id:        str
    chunk_text:      str
    source_block_id: str
    source_page:     int
    entities:        List[Entity]
    relations:       List[Relation]

@dataclass
class DedupedExtractionResult:
    document_id:       str
    entities:          List[Entity]          # canonical entities only
    relations:         List[Relation]        # updated to canonical IDs
    chunk_extractions: List[ChunkExtraction] # raw, for traceability
    dedup_log:         List[Dict]            # {"merged": ["Jon","John"], "canonical": "John"}
    metadata:          Dict[str, Any]
```

---

## 5. Format Detection Logic (detector.py)

`detect_format(source_name, raw_bytes, raw_text)` returns `(format_type, mime_type, encoding)`.

**Priority order:**
1. If `raw_text` is passed in directly → check for mixed format markers; return `"text"` or `"mixed"`.
2. File **extension mapping** (most reliable on Windows):
   - `.csv/.json/.html/.xml/.md` → `"structured"`
   - `.txt` → `"text"`
   - `.pdf/.docx/.xlsx/.xls` → `"document"`
   - `.jpg/.png/.gif/.bmp/.tiff` → `"image"`
3. `mimetypes.guess_type` fallback.
4. Content-based sniffing (try decode, check printable).
5. Return `"unknown"` → triggers explicit error in `batch_processor`.

**Mixed content detection** (`_has_mixed_formats`): looks for embedded format markers (e.g., JSON objects inside XML, code blocks inside HTML, etc.) in the decoded text.

---

## 6. Four Chunking Strategies (strategies.py)

All strategies accept `List[Segment]` where `Segment = (text: str, source_metadata: dict)` and return `List[Chunk]`.

### A — Line (finest)
- Split on `\n`, then on sentence boundaries (`.`, `!`, `?` via regex).
- Headings (ALL CAPS, `##`, `Chapter/Section N`) stay as their own chunk.

### B — Paragraph/Section (medium)
- Split on `\n\n` (blank lines) → individual paragraphs.
- Detect headings via `_SECTION_HEADER` regex (Markdown `##`, `1. Title`, `ALL CAPS`, `Chapter …`).
- Group paragraphs between headings as one section chunk.
- **Three fallback tiers for headingless documents:**
  1. Greedy paragraph packing with token count guard + overlap carry-forward.
  2. spaCy sentence splitting if no blank lines.
  3. Sliding token window as last resort (sets `metadata.fallback = "sliding_window"`).

### C — Page
- One chunk = one `Page` object from `BinaryDocument`.
- Blocks sorted by `(bbox.y0, bbox.x0)` for correct reading order.

### D — Context / BERTopic (semantic)
- Embed all paragraphs with `sentence-transformers` (`BAAI/bge-m3`, 1024-dim).
- UMAP dim reduction → HDBSCAN clustering → BERTopic topic labels.
- Paragraphs in the same topic cluster → same `ContextGroup` (merged into one SLM-ready chunk).
- Outliers (topic `= -1`) → each becomes its own chunk.
- Requires ≥ 4 paragraphs; falls back to Paragraph strategy for smaller inputs.
- Silhouette coherence score stored per group and overall.
- Max 8 context groups per document (`_MAX_CONTEXT_GROUPS = 8`).

---

## 7. VLM Layout Detection (vlm/ module)

### Model: PaliGemma 2
| Variant | VRAM (BF16) | Resolution |
|---|---|---|
| `google/paligemma2-3b-pt-224` | ~6 GB | 224 px |
| `google/paligemma2-10b-pt-448` | ~20 GB | 448 px (better spatial precision) |

**Prompt** (used in `paligemma_adapter.py`):
```
System: You are a document layout analysis model.
User:   <image>
        Identify all visually distinct content blocks in this document.
        For each block return a JSON object:
        {"label": "<semantic_label>", "bbox": [x0, y0, x1, y1]}
        Return a JSON array of all blocks.
```

**Integration point:** `binary_handler.py` calls `layout_detector.run_layout_detection(doc)` BEFORE OCR. Returns `List[LayoutBlock]` which is passed into the updated OCR dispatcher.

**Edge cases:**
- Low confidence / blank page → fall back to whole-page as single block.
- PDF pages rasterised at 150–300 DPI via `PyMuPDF` (`fitz`) before VLM call.
- **Multi-column text-only PDFs:** Use `page.get_text("rawdict")` to get span `x0` coordinates. Gap analysis (gaps > 15% of page width) detects columns. Sort spans top-to-bottom within each column, then left-to-right across columns. Only invoke VLM if ≥ 3 columns OR mixed image+text blocks detected.

---

## 8. OCR Pipeline (ocr/ module)

### Chandra OCR
- External CLI / gRPC service (`CHANDRA_OCR_ENDPOINT`, default `localhost:50051`).
- Phase 2 adds `run_chandra_on_block(cropped_bytes, block_id, block_label)`:
  - Crops page image to each `LayoutBlock.bbox` using `PIL.Image.crop()`.
  - Runs OCR on the crop.
  - Chandra assigns a block-level title (heading/section detection built in).
  - Returns: `{block_id, title, label, text, words: [{text, bbox, conf}], confidence}`.
- Fallback: `PyPDF2` text extraction → `PyMuPDF` for scanned PDFs.

---

## 9. OCR Spell Correction (Phase 2 — postprocessing/spell_correction.py)

Three-stage pipeline applied per `Block.raw_text`:

**Stage A — Regex OCR noise patterns (no GPU, fast)**
- Confidence-gated: only applies to tokens with `conf < 0.85`.
- Patterns: `0→O`, `1→I`, `rn→m`, camelCase OCR merge splits, etc.

**Stage B — SymSpell (CPU, milliseconds)**
- Edit distance 1–2 for unknown words.
- Skips tokens in proper-noun whitelist (populated from prior SLM entity lists).
- Library: `symspellpy`.

**Stage C — neuspell BERT correction (GPU, ambiguous cases only)**
- Triggered only when SymSpell returns multiple equal-frequency candidates.
- Model: `bert-base-cased`, ~0.5 GB VRAM.
- Library: `neuspell`.

Each `Block` stores both `raw_text` and `corrected_text`. Downstream uses `corrected_text`.

---

## 10. SLM Entity & Relation Extraction (Phase 2 — postprocessing/slm_extractor.py)

### Model choice (24 GB budget)
| Model | VRAM | Recommended |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | ✅ Primary |
| `microsoft/Phi-3.5-mini-instruct` | ~7.6 GB | Lightweight alt |
| `mistralai/Mistral-7B-Instruct-v0.3` | ~14 GB | Alt |

**Fine-tuning:** LoRA (rank 16) on `(chunk_text, expected_json)` pairs using `trl` SFT Trainer.

**Entity types:** `PERSON, ORG, PRODUCT, LOCATION, DATE, EVENT, ROLE, SKILL`  
**Relation types:** `WORKS_AT, FOUNDED_BY, LOCATED_IN, COLLABORATED_WITH, HAS_SKILL, PUBLISHED_ON, MENTIONS, PART_OF`

**Loading pattern:**
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
```
Load once, reuse across all documents. Use `vllm` or `text-generation-inference` for high-throughput batch serving.

---

## 11. Fuzzy Deduplication (Phase 2 — postprocessing/deduplicator.py)

**Problem:** Same entity appears as `"Jon"/"John"`, `"Micr0s0ft"/"Microsoft"`, `"Microsoft Corp"/"Microsoft Corporation"`.

**Algorithm:**
1. **Blocking:** group by `entity_type + first_3_chars_lowercase` → reduces O(n²) to O(n log n).
2. **Scoring:** `rapidfuzz.fuzz.token_set_ratio` per pair (handles reordering: "John Smith" vs "Smith, John").
3. **Merge:** Union-Find DSU; canonical = most frequent text in merged set.
4. **Update:** All relation `head`/`tail` IDs updated to canonical. Self-referencing relations removed.

**Thresholds:**
| Entity type | Threshold |
|---|---|
| Short words ≤ 6 chars | 92 |
| Person names | 88 |
| Organisation names | 85 |
| Default | 88 |

---

## 12. VRAM Budget (24 GB GPU)

| Component | VRAM |
|---|---|
| PaliGemma 2 3B (VLM) — BF16 | ~6 GB |
| neuspell BERT (spell correction) | ~0.5 GB |
| BGE-M3 embeddings (BERTopic) | ~2.2 GB |
| Qwen2.5-7B SLM (extraction) — BF16 | ~14 GB |
| **Total** | **~22.7 GB** |
| **Headroom** | **~1.3 GB** |

> INT8 quantisation of PaliGemma 2 3B via `bitsandbytes` saves ~3 GB if needed.  
> Sequential processing: run VLM → OCR → correction → chunk → SLM per page; call `torch.cuda.empty_cache()` between VLM and SLM if VRAM is tight.

---

## 13. Streamlit UI (ui/streamlit_app_unified.py)

**Current tabs (Phase 1):**
1. 📋 Extracted Content
2. 🔍 Technical Details
3. 📦 Export

**Phase 2 additions:**
- **Sidebar panel "🧩 Extraction Settings":**
  - Chunking strategy radio (`line | paragraph | section | context`)
  - Max tokens per chunk slider (64–2048, default 512)
  - Chunk overlap slider (0–256, default 64)
  - Deduplication threshold slider (default 88%)
  - Checkboxes: Spell Correction / Entity Extraction / Relation Extraction / Export Entity Graph (GraphML)

- **New tabs:**
  4. 🏷️ Entities & Relations — entities table, relations table, `pyvis` interactive graph
  5. 🧩 Chunks — chunk list with source metadata (page, block, VLM label, entities found)

---

## 14. What Is Already Implemented (Phase 1 — DONE ✅)

| Component | Status |
|---|---|
| MIME detection + routing (detector.py / router.py) | ✅ |
| HTML/CSV/JSON/Markdown parsing | ✅ |
| Mixed-content format separation | ✅ |
| DOCX handler (python-docx) | ✅ |
| XLSX handler (openpyxl) | ✅ |
| PDF OCR (Chandra CLI + PyPDF2 fallback) | ✅ Partial |
| Image OCR (Chandra CLI) | ✅ Partial |
| `BinaryDocument → Page → Region` schema | ✅ |
| Streamlit UI (single / batch / text modes) | ✅ |
| Chunking engine (all 4 strategies, schemas) | ✅ |
| VLM module scaffold (`vlm/` directory + files) | ✅ Structure only |
| Batch processing + JSON output | ✅ |

---

## 15. What Is Still Planned (Phase 2 — IN PROGRESS 🔄)

### Sprint 1 — VLM Integration
- [ ] `paligemma_adapter.py`: load model, inference, parse JSON layout output
- [ ] `layout_detector.py`: PDF rasterisation + per-page VLM call + crop logic
- [ ] Update `binary_handler.py` to call VLM before OCR
- [ ] Update `binary_schema.py`: extend `Page` with `blocks: List[Block]`
- [ ] Unit tests with mocked VLM output

### Sprint 2 — Spell Correction
- [ ] `postprocessing/spell_correction.py` (Stages A, B, C)
- [ ] Add `raw_text`/`corrected_text` to `Block`
- [ ] Wire into `binary_handler.py` post-OCR

### Sprint 3 — Chunking Wire-up
- [ ] Connect chunker to `postprocessing/pipeline.py`
- [ ] Unit tests per strategy + token limit compliance

### Sprint 4 — SLM Extraction
- [ ] Fine-tune Qwen2.5-7B with LoRA on domain NER/RE dataset
- [ ] `postprocessing/slm_extractor.py`: load model, batch inference, JSON parsing + retry
- [ ] `postprocessing/extraction_schema.py`: `Entity`, `Relation`, `ChunkExtraction`

### Sprint 5 — Fuzzy Dedup
- [ ] `postprocessing/deduplicator.py`: blocking + RapidFuzz + Union-Find
- [ ] `DedupedExtractionResult` schema

### Sprint 6 — UI & End-to-End Integration
- [ ] Sidebar extraction settings panel
- [ ] Entities & Relations tab + pyvis graph
- [ ] Chunks tab with provenance
- [ ] `postprocessing/pipeline.py` orchestrating all 6 steps
- [ ] End-to-end test with LinkedIn screenshot + multi-page PDF

### Sprint 7 — Testing & Optimisation
- [ ] Full test suite for Phase 2 modules
- [ ] VRAM profiling
- [ ] INT8 quantisation fallback for memory-constrained environments

---

## 16. Key Design Decisions (Rationale for LLM)

| Decision | Rationale |
|---|---|
| Extension-first MIME detection | `mimetypes` is unreliable on Windows; `.csv` guesses as `text/plain`. Extension mapping is deterministic. |
| Chandra OCR as primary | GPU-accelerated, handles complex layouts natively; falls back to PyPDF2/PyMuPDF for text-only PDFs. |
| PaliGemma 2 for layout | Open-weights VLM that fits in 6 GB (3B) or 20 GB (10B), outputs structured JSON bboxes. |
| BERTopic for context chunking | Semantic grouping prevents context fragmentation for SLM; related paragraphs from different pages land in the same chunk. |
| RapidFuzz for dedup | C++-backed, 100× faster than pure Python; `token_set_ratio` correctly handles name reordering. |
| Union-Find for entity merge | O(α(n)) near-linear merge with path compression; simpler than full graph clustering for this scale. |
| `raw_text` + `corrected_text` in Block | Keeps original OCR output for audit while feeding corrected text to downstream models. |
| Backwards-compat `regions` property on Page | Phase 1 code using `page.regions` continues to work after Phase 2 schema extension. |
| LoRA fine-tuning Qwen2.5-7B | Training full 7B is impractical on 24 GB; LoRA rank 16 adds ~25 M trainable params. Domain fine-tuning on (chunk, json) pairs improves structured JSON output reliability. |

---

## 17. Environment & Dependencies

### Core
```
streamlit>=1.50.0
pandas
chardet
beautifulsoup4
lxml
python-docx==1.2.0
openpyxl==3.1.5
Pillow
PyPDF2
PyMuPDF
chandra-ocr>=0.1.8
```

### Phase 2 AI/ML
```
transformers>=4.46.0
accelerate>=0.30.0
bitsandbytes>=0.43.0
torch>=2.2.0
sentencepiece
huggingface_hub>=0.23.0
sentence-transformers>=3.0.0
bertopic>=0.16.4
umap-learn>=0.5.6
hdbscan>=0.8.38
spacy>=3.7.0       # + python -m spacy download en_core_web_sm
symspellpy>=6.7.7
neuspell>=1.0.0
rapidfuzz>=3.6.0
networkx>=3.2.0
pyvis>=0.3.2
```

### Environment Setup
```bash
git clone https://github.com/garvitsachdevaa/ETL-PIPELINE.git
cd ETL-PIPELINE
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r etl_pipeline/requirements.txt
```

### Key Env Vars
| Var | Default | Purpose |
|---|---|---|
| `CHANDRA_OCR_ENDPOINT` | `localhost:50051` | Chandra gRPC endpoint |
| `HF_TOKEN` | — | HuggingFace token for gated models (PaliGemma 2) |
| `LOG_LEVEL` | `INFO` | Python logging |
| `ETL_MAX_WORKERS` | `4` | Batch processor parallelism |
| `BERTOPIC_RANDOM_SEED` | `42` | BERTopic reproducibility |

---

## 18. Quick Stage-to-Stage Reference

```
Stage                Input                          Output
─────────────────────────────────────────────────────────────────────────────
loader.py            file path / bytes              DocumentObject
detector.py          DocumentObject                 (format_type, mime_type, encoding)
router.py            format_type                    handler name string
text_handler.py      DocumentObject (text)          TextDocument {sections}
binary_handler.py    DocumentObject (binary)        BinaryDocument {pages→blocks→regions}
vlm/layout_detector  BinaryDocument (image/PDF)     List[LayoutBlock] with bbox + label
ocr/chandra_adapter  LayoutBlock + cropped image    Block {title, raw_text, words, conf}
spell_correction.py  Block.raw_text                 Block.corrected_text
chunking/chunker.py  TextDocument or BinaryDocument ChunkingResult {List[Chunk]}
slm_extractor.py     Chunk.text                     ChunkExtraction {entities, relations}
deduplicator.py      List[ChunkExtraction]          DedupedExtractionResult
UI / batch export    DedupedExtractionResult        Streamlit display / JSON file
```

---

*context.md last updated: February 27, 2026*
