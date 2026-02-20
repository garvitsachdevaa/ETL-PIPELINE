# ETL Pipeline — Phase 2 Full Implementation Planner

> **Date:** February 19, 2026  
> **GPU:** 24 GB VRAM (HuggingFace Inference Endpoint / Local)  
> **Branch:** `dev_branch`

---

## Table of Contents

1. [Current System Summary](#1-current-system-summary)
2. [Full Architecture — Phase 2](#2-full-architecture--phase-2)
3. [Phase 1 — VLM Layout Detection (PaliGemma)](#3-phase-1--vlm-layout-detection-paligemma)
4. [Phase 2 — Block-Aware OCR (Chandra)](#4-phase-2--block-aware-ocr-chandra)
5. [Phase 3 — OCR Spell Correction](#5-phase-3--ocr-spell-correction)
6. [Phase 4 — Chunking Engine](#6-phase-4--chunking-engine)
7. [Phase 5 — SLM Entity & Relation Extraction](#7-phase-5--slm-entity--relation-extraction)
8. [Phase 6 — Fuzzy Deduplication](#8-phase-6--fuzzy-deduplication)
9. [Phase 7 — UI Chunking Controls (Streamlit)](#9-phase-7--ui-chunking-controls-streamlit)
10. [Schema Extensions](#10-schema-extensions)
11. [GPU / Infrastructure Plan](#11-gpu--infrastructure-plan)
12. [New File Structure](#12-new-file-structure)
13. [New Dependencies](#13-new-dependencies)
14. [Implementation Roadmap (Sprints)](#14-implementation-roadmap-sprints)

---

## 1. Current System Summary

```
Ingest (loader.py)
  ↓ detect MIME/format
Route (router.py)
  ├─ text/structured  → text_handler.py  → TextDocument  (sections)
  ├─ image/document   → binary_handler.py → BinaryDocument (pages → regions)
  └─ mixed            → mixed_handler.py  → TextDocument  (multi-format sections)
```

**What is already working:**
| Component | Status |
|---|---|
| MIME detection + routing | ✅ |
| HTML/CSV/JSON/Markdown parsing (BeautifulSoup etc.) | ✅ |
| Mixed-content format separation | ✅ |
| DOCX handler (python-docx) | ✅ |
| XLSX handler (openpyxl) | ✅ |
| PDF OCR (Chandra CLI + PyPDF2 fallback) | ✅ Partial |
| Image OCR (Chandra CLI) | ✅ Partial |
| `BinaryDocument → Page → Region` schema | ✅ |
| Streamlit UI (single/batch/text) | ✅ |

**What is missing (Phase 2 targets):**
- VLM layout detection (block boundaries with semantic labels)
- Block-aware, titled OCR
- OCR-specific spell correction
- Multi-strategy chunking engine
- SLM entity & relation extraction
- Fuzzy entity deduplication
- UI chunking controls

---

## 2. Full Architecture — Phase 2

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ETL PIPELINE — PHASE 2                         │
└────────────────────────────────────────────────────────────────────────┘

 UPLOAD (PDF / Image / Text-with-Images)
         │
         ▼
 ┌───────────────┐
 │  Ingest +     │  loader.py  →  detector.py  →  DocumentObject
 │  Detect       │
 └───────┬───────┘
         │
         ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                    binary_handler.py                          │
 │                                                               │
 │   ┌─────────────────────────────────────────────────────┐    │
 │   │  STEP 1 — VLM Layout Detection                      │    │
 │   │                                                     │    │
 │   │  handlers/vlm/paligemma_adapter.py                  │    │
 │   │  • Input : full-page image (PIL / bytes)            │    │
 │   │  • Model : PaliGemma 2 (3B or 10B)                  │    │
 │   │  • Prompt: "Detect all document layout blocks.      │    │
 │   │             Return each block as JSON with          │    │
 │   │             {label, bbox:[x0,y0,x1,y1]}"            │    │
 │   │  • Output: List[LayoutBlock]                        │    │
 │   │            ┌──────────────────────────────┐         │    │
 │   │            │ label: "user_profile"        │         │    │
 │   │            │ bbox : [0, 0, 250, 800]      │         │    │
 │   │            ├──────────────────────────────┤         │    │
 │   │            │ label: "feed_posts"          │         │    │
 │   │            │ bbox : [250, 0, 750, 800]    │         │    │
 │   │            ├──────────────────────────────┤         │    │
 │   │            │ label: "promotions"          │         │    │
 │   │            │ bbox : [750, 0, 1000, 800]   │         │    │
 │   │            └──────────────────────────────┘         │    │
 │   └──────────────────────┬──────────────────────────────┘    │
 │                          │  bounding boxes per block         │
 │   ┌──────────────────────▼──────────────────────────────┐    │
 │   │  STEP 2 — Block-Aware Chandra OCR                   │    │
 │   │                                                     │    │
 │   │  handlers/ocr/chandra_adapter.py (updated)          │    │
 │   │  • Crop image to each LayoutBlock bbox              │    │
 │   │  • Run Chandra OCR on each cropped region           │    │
 │   │  • Chandra assigns a title to each block            │    │
 │   │  • Output: List[OcrBlock] with text + title + bbox  │    │
 │   └──────────────────────┬──────────────────────────────┘    │
 │                          │                                    │
 │   ┌──────────────────────▼──────────────────────────────┐    │
 │   │  STEP 3 — OCR Spell Correction                      │    │
 │   │                                                     │    │
 │   │  postprocessing/spell_correction.py                 │    │
 │   │  • SymSpell (fast edit-distance lookup)             │    │
 │   │  • Custom OCR noise patterns (0→O, 1→I, rn→m …)    │    │
 │   │  • Context-aware BERT correction for ambiguous      │    │
 │   │    cases (neuspell on 24 GB GPU)                    │    │
 │   │  • Preserves proper nouns in a whitelist            │    │
 │   └──────────────────────┬──────────────────────────────┘    │
 └─────────────────────────┬┘                                    │
                           │   EnrichedBinaryDocument            │
                           ▼                                     │
 ┌─────────────────────────────────────────────────────────────────┐
 │  POST-PROCESSING PIPELINE (postprocessing/pipeline.py)         │
 │                                                                 │
 │  ┌───────────────────────────────────────────────────────────┐  │
 │  │  STEP 4 — Chunking Engine  (user-controlled via UI)       │  │
 │  │                                                           │  │
 │  │  postprocessing/chunker.py                                │  │
 │  │                                                           │  │
 │  │  Strategy A — Sentence / Line                             │  │
 │  │    Split on . ! ? or \n using spaCy sentence boundary    │  │
 │  │                                                           │  │
 │  │  Strategy B — Paragraph / Section                        │  │
 │  │    Split on \n\n, detect headings via regex / font size  │  │
 │  │                                                           │  │
 │  │  Strategy C — Page                                        │  │
 │  │    One chunk = one page (already in BinaryDocument)      │  │
 │  │                                                           │  │
 │  │  Strategy D — Context (BERTopic)                         │  │
 │  │    postprocessing/bertopic_chunker.py                     │  │
 │  │    1. Embed paragraphs → sentence-transformers           │  │
 │  │       (all-MiniLM-L6-v2 or BGE-M3 on GPU)               │  │
 │  │    2. UMAP dimensionality reduction                      │  │
 │  │    3. HDBSCAN clustering → topic groups                  │  │
 │  │    4. Paragraphs in same topic cluster → same chunk      │  │
 │  │    BERTopic ensures semantically coherent chunks         │  │
 │  └───────────────────────┬───────────────────────────────────┘  │
 │                          │   List[Chunk]                         │
 │  ┌───────────────────────▼───────────────────────────────────┐  │
 │  │  STEP 5 — SLM Entity & Relation Extraction               │  │
 │  │                                                           │  │
 │  │  postprocessing/slm_extractor.py                         │  │
 │  │  • Fine-tuned SLM (Qwen2.5-7B / Phi-3.5 / Mistral-7B)   │  │
 │  │  • Input : each Chunk                                    │  │
 │  │  • Output: structured JSON per chunk                     │  │
 │  │    {                                                      │  │
 │  │      "entities": [                                        │  │
 │  │        {"id": "e1", "text": "Microsoft",                  │  │
 │  │         "type": "ORG", "span": [12, 21]},  ...           │  │
 │  │      ],                                                   │  │
 │  │      "relations": [                                       │  │
 │  │        {"head": "e1", "tail": "e2",                       │  │
 │  │         "type": "FOUNDED_BY"},  ...                       │  │
 │  │      ]                                                    │  │
 │  │    }                                                      │  │
 │  └───────────────────────┬───────────────────────────────────┘  │
 │                          │   List[ChunkExtractions]              │
 │  ┌───────────────────────▼───────────────────────────────────┐  │
 │  │  STEP 6 — Fuzzy Deduplication                            │  │
 │  │                                                           │  │
 │  │  postprocessing/deduplicator.py                          │  │
 │  │  • Aggregate all entities across all chunks              │  │
 │  │  • Blocking: group by first 3 chars + entity type       │  │
 │  │  • RapidFuzz: token_set_ratio similarity per pair       │  │
 │  │  • Threshold ≥ 88% → merge into canonical entity        │  │
 │  │  • Update all relation heads/tails to canonical IDs     │  │
 │  │  • Jon ↔ John, Micr0s0ft ↔ Microsoft → same entity     │  │
 │  └───────────────────────┬───────────────────────────────────┘  │
 │                          │                                       │
 └──────────────────────────┼───────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  ExtractionResult       │
              │  • entities (deduped)   │
              │  • relations            │
              │  • chunks (with source) │
              │  • raw OCR blocks       │
              └─────────────────────────┘
                            │
                            ▼
                     Streamlit UI + JSON Export
```

---

## 3. Phase 1 — VLM Layout Detection
(PaliGemma)

### What it does
Given a **full-page image** (or a PDF page rasterised to an image), PaliGemma 2 returns a list of **visual blocks** with:
- Semantic label (e.g., `"user_profile"`, `"feed_posts"`, `"advertisement"`, `"table"`, `"figure"`, `"header"`)
- Bounding box in pixel coordinates `[x0, y0, x1, y1]`

### Model Choice
| Model | VRAM (BF16) | Recommended Use |
|---|---|---|
| `google/paligemma2-3b-pt-224` | ~6 GB | Fast layout detection, good for dense documents |
| `google/paligemma2-10b-pt-448` | ~20 GB | Higher resolution, better spatial precision |

> **With 24 GB**: Use **PaliGemma 2 10B @ 448px** for images. For batch PDFs, quantise to INT8 (~10 GB) and run concurrently with SLM.

### Prompt Engineering
```
System: You are a document layout analysis model.
User:   <image>
        Identify all visually distinct content blocks in this document.
        For each block return a JSON object:
        {"label": "<semantic_label>", "bbox": [x0, y0, x1, y1]}
        Where bbox values are pixel coordinates.
        Return a JSON array of all blocks.
```

### Output Schema — `LayoutBlock`
```python
@dataclass
class LayoutBlock:
    block_id: str           # uuid4
    label: str              # VLM-assigned semantic label
    bbox: List[int]         # [x0, y0, x1, y1]
    confidence: float       # VLM logit confidence if available
    page_number: int
```

### New Files
- `etl_pipeline/handlers/vlm/__init__.py`
- `etl_pipeline/handlers/vlm/paligemma_adapter.py` — loads model, runs inference, parses JSON output
- `etl_pipeline/handlers/vlm/layout_detector.py` — crops PIL image per block, calls adapter
- `etl_pipeline/handlers/vlm/block_schema.py` — `LayoutBlock` dataclass

### Integration Point
`binary_handler.py` will call `layout_detector.run_layout_detection(doc)` **before** calling OCR. The returned `List[LayoutBlock]` is passed into the updated OCR dispatcher.

### Edge Cases
- If PaliGemma cannot detect blocks (low confidence / blank page) → fall back to whole-page as single block
- PDFs: each page is rasterised using **PyMuPDF (fitz)** at 150–300 DPI before passing to VLM
- **Text-only PDFs with selectable text and multi-column layout:**
  PyMuPDF's `page.get_text("rawdict")` extracts individual text spans with their `x0, y0` coordinates. A naive `get_text("text")` call reads spans top-to-bottom in DOM order and will **interleave columns** (line 1 col-A + line 1 col-B → line 2 col-A + line 2 col-B…), producing garbled output.

  **Detection:** After extracting spans, inspect the x-coordinate distribution. If the page contains two or more distinct horizontal clusters of `x0` values (detected via a simple gap analysis: sort all `x0` values, find gaps > 15% of page width), the page is flagged as **multi-column**.

  **Column-aware reading order algorithm:**
  ```python
  # 1. Cluster spans into columns by x0 gap analysis
  columns = cluster_spans_by_x0(spans, page_width, gap_threshold=0.15)
  # e.g. columns = [[col_A_spans], [col_B_spans]]

  # 2. Within each column: sort spans top-to-bottom by y0
  for col in columns:
      col.sort(key=lambda s: s["y0"])

  # 3. Read columns left-to-right
  ordered_text = " ".join(span["text"] for col in columns for span in col)
  ```
  This preserves the correct reading order: finish column A entirely before starting column B.

  **When to still invoke VLM:** If the column gap analysis detects ≥ 3 columns, or if mixed image+text blocks are present within the selectable-text PDF (detected by checking `page.get_images()`), route through VLM anyway — the spatial complexity is high enough that VLM block detection is worth the cost.

---

## 4. Phase 2 — Block-Aware OCR (Chandra)

### Updated OCR Flow

```
LayoutBlock list
    │
    ├─ For each block:
    │     crop_image(page_image, block.bbox)   ← PIL.Image.crop()
    │     chandra_ocr(cropped_image)           ← Chandra CLI per crop
    │     → text, word_bboxes, confidence
    │
    └─ Chandra assigns block-level title
          (uses its internal heading/section detection)
```

### Updated `chandra_adapter.py`
Add `run_chandra_on_block(cropped_bytes, block_id, block_label)` alongside the existing `run_chandra_cli`. Returns:
```python
{
  "block_id": "...",
  "title":    "User Profile Section",   # Chandra-assigned
  "label":    "user_profile",           # VLM-assigned (passed through)
  "text":     "John Doe\nSoftware Engineer at ...",
  "words":    [{"text": "John", "bbox": [...], "conf": 0.98}, ...],
  "confidence": 0.96
}
```

### Updated `binary_schema.py`
Extend the existing schema:
```python
@dataclass
class Block:
    block_id:   str
    title:      str              # Chandra-assigned title
    label:      str              # VLM semantic label
    bbox:       List[int]        # from VLM
    regions:    List[Region]     # existing Region objects (word-level)
    confidence: float
    metadata:   Dict[str, Any]

@dataclass
class Page:
    page_id:    str
    page_number: int
    blocks:     List[Block]      # replaces flat regions
    metadata:   Dict[str, Any]
```
> **Backwards compatibility**: keep `regions` as a flattened property computed from `[r for b in blocks for r in b.regions]`

---

## 5. Phase 3 — OCR Spell Correction

### Problem
Chandra OCR on scanned/screenshot images may produce:
- Character substitution: `0` → `O`, `1` → `I`, `rn` → `m`, `cl` → `d`
- Word breaks: `Mi crosoft` instead of `Microsoft`
- Missing spaces: `MicrosoftCorporation`
- Low-confidence characters on degraded scans

### Two-Stage Correction Strategy

#### Stage A — Rule-Based OCR Noise Patterns (Fast, No GPU)
Located in `postprocessing/spell_correction.py`

```python
OCR_SUBSTITUTIONS = [
    (r'\b0(?=[A-Za-z])', 'O'),   # 0pen → Open
    (r'(?<=[A-Za-z])0\b', 'o'),  # hell0 → hello
    (r'\b1(?=[A-Za-z])', 'I'),   # 1BM → IBM
    (r'(?<=[A-Za-z])1\b', 'l'),  # fee1 → feel
    (r'rn', 'm'),                 # rnodern → modern (context-gated)
    (r'(?<!\s)([A-Z])', r' \1'), # camelCase split for OCR merges
    # ... extend per domain
]
```

Apply **only to tokens below OCR confidence threshold** (e.g., `conf < 0.85`).

#### Stage B — SymSpell Dictionary Correction (CPU, Milliseconds)
- Uses frequency-ranked dictionary (English + domain-specific terms)
- Edit distance 1–2 for unknown words
- Skips tokens in a **proper noun whitelist** (populated from the SLM entity list of previous docs)
- Library: `symspellpy`

#### Stage C — Context-Aware Neural Correction (GPU, For Ambiguous Cases)
- Use **neuspell** (BERT-based context spell checker)
- Only triggered when SymSpell returns multiple candidates with equal frequency
- With 24 GB GPU: runs `bert-base-cased` model in parallel with VLM (VLM uses ~6 GB, neuspell uses ~0.5 GB)
- Library: `neuspell`

### Correction Pipeline per Block
```
raw_text_from_chandra
    → Stage A (regex patterns, confidence-gated)
    → Stage B (SymSpell, unknown word lookup)
    → Stage C (neuspell, only if ambiguous tokens remain)
    → corrected_text
```

### Output
Each `Block` stores both `raw_text` and `corrected_text`. Downstream chunking and SLM use `corrected_text`.

---

## 6. Phase 4 — Chunking Engine

### User Controls (exposed in Streamlit UI)
```
Chunking Strategy:  ○ Sentence/Line   ○ Paragraph/Section   ○ Page   ● Context (BERTopic)
Max Chunk Tokens:   [512]   (slider, 64–2048)
Chunk Overlap:      [64]    (slider, 0–256, for sliding window)
```

### Strategy A — Sentence / Line
```python
# Using spaCy sentence boundary detection (en_core_web_sm)
import spacy
nlp = spacy.load("en_core_web_sm")
sentences = [sent.text for sent in nlp(block.corrected_text).sents]
```
- Splits after every `.`, `!`, `?` using spaCy's sentence segmenter
- Packs sentences into chunks until `max_chunk_tokens` is reached
- Overlap: last `N` tokens of previous chunk prepended to next

### Strategy B — Paragraph / Section
```python
# Split on blank lines first, then detect heading markers
paragraphs = re.split(r'\n{2,}', text)
# Detect section headings: ALL CAPS lines, lines ending with ':', numbered headers
# e.g. "INTRODUCTION", "3. Methodology", "Results:"
headings = [p for p in paragraphs if is_heading(p)]
```
- Groups text between detected headings as a single section chunk
- If section > `max_chunk_tokens`: recursively split into paragraph sub-chunks

**When no headings are found (headingless document):**

This is common in OCR output from screenshots, scanned letters, social media posts, or plain prose. The strategy falls back gracefully through three tiers:

1. **Tier 1 — Blank-line paragraph packing:**
   The `\n{2,}` split already gives individual paragraphs. These are packed greedily into chunks up to `max_chunk_tokens`, with `chunk_overlap` tokens carried forward from the previous chunk. No heading required — paragraph boundaries _are_ the chunk boundaries.
   ```python
   chunks, current, current_tokens = [], [], 0
   for para in paragraphs:
       para_tokens = token_count(para)
       if current_tokens + para_tokens > max_chunk_tokens and current:
           chunks.append(" ".join(current))
           # carry overlap: keep last N tokens of current chunk
           current = [get_last_n_tokens(" ".join(current), overlap_tokens)]
           current_tokens = token_count(current[0])
       current.append(para)
       current_tokens += para_tokens
   if current:
       chunks.append(" ".join(current))
   ```

2. **Tier 2 — Single long paragraph (wall of text, no blank lines):**
   If the entire block is one paragraph (no `\n\n` found), fall back to spaCy sentence boundary splitting (same as Strategy A) and then pack sentences into `max_chunk_tokens`-sized windows with overlap. This avoids creating a single enormous chunk that overflows the SLM context window.

3. **Tier 3 — No sentence boundaries either (e.g., table cells, list items, address blocks):**
   Fall back to a sliding token window: split by whitespace tokens, slide a window of `max_chunk_tokens` with `chunk_overlap` stride. Each window becomes one chunk. A `metadata` flag `{"fallback": "sliding_window"}` is set on these chunks so the SLM extractor knows context may be less coherent.

In all three tiers, each output `Chunk` stores `source_block_id`, `source_page`, and `chunk_strategy` so the results tab in the UI can indicate which tier was used per chunk.

### Strategy C — Page
- Directly uses the `Page` objects from `BinaryDocument`
- Each `Page` → one chunk, concatenating all `Block.corrected_text` in reading order
- Reading order: sort blocks by `(bbox.y0, bbox.x0)` (top-to-bottom, left-to-right)

### Strategy D — Context / BERTopic
Located in `postprocessing/bertopic_chunker.py`

**Full Algorithm:**
```
1. Input: List of all paragraphs across all blocks (with source metadata)

2. Embed each paragraph:
   model = SentenceTransformer("BAAI/bge-m3")   # 24GB → fits on GPU
   embeddings = model.encode(paragraphs, batch_size=64)
   # bge-m3: 1024-dim, multilingual, excellent for domain text

3. BERTopic pipeline:
   from bertopic import BERTopic
   from umap import UMAP
   from hdbscan import HDBSCAN

   umap_model  = UMAP(n_components=5, min_dist=0.0, metric='cosine')
   hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean',
                            cluster_selection_method='eom',
                            prediction_data=True)
   topic_model = BERTopic(umap_model=umap_model,
                          hdbscan_model=hdbscan_model,
                          verbose=False)
   topics, probs = topic_model.fit_transform(paragraphs, embeddings)

4. Group paragraphs by topic:
   topic_groups = defaultdict(list)
   for para, topic in zip(paragraphs, topics):
       topic_groups[topic].append(para)
   # topic == -1 are outliers → each becomes its own chunk

5. Pack each topic group into chunks respecting max_chunk_tokens
   (sorted by original document order within the group)
```

**Why BERTopic for chunking?**
- LinkedIn screenshot: posts about "Product Launch" cluster together even if scattered across 3 pages
- The SLM sees all related context in one chunk → better relation extraction
- Reduces hallucination from context fragmentation

---

## 7. Phase 5 — SLM Entity & Relation Extraction

### Model Choice (24 GB GPU)

| Model | VRAM (BF16) | Speed | Quality |
|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | Fast | ⭐⭐⭐⭐⭐ |
| `microsoft/Phi-3.5-mini-instruct` (3.8B) | ~7.6 GB | Very Fast | ⭐⭐⭐⭐ |
| `mistralai/Mistral-7B-Instruct-v0.3` | ~14 GB | Fast | ⭐⭐⭐⭐ |
| `meta-llama/Llama-3.2-3B-Instruct` | ~6 GB | Very Fast | ⭐⭐⭐ |

> **Recommendation**: Fine-tune **Qwen2.5-7B** on your domain data using LoRA (rank 16) on HuggingFace with 24 GB. During inference: VLM (PaliGemma 3B, ~6GB) + SLM (Qwen 7B, ~14GB) = ~20GB total — fits comfortably.

### Fine-Tuning Strategy
- **Framework**: `trl` (SFT Trainer) + `peft` (LoRA)
- **Dataset format**: Each training example = `(chunk_text, expected_json_output)`
- **LoRA config**: `r=16`, `lora_alpha=32`, `target_modules=["q_proj","v_proj"]`
- **Training**: 3–5 epochs, `bf16=True`, gradient checkpointing
- **HuggingFace Space**: Use a Space with A100/H100 (40/80 GB) for training, 24 GB for inference

### Extraction Prompt Template
```
System: You are a named entity and relation extraction model.
        Extract all entities and their relationships from the given text.
        Return ONLY valid JSON. No explanation.
        Entity types: PERSON, ORG, PRODUCT, LOCATION, DATE, EVENT, ROLE, SKILL
        Relation types: WORKS_AT, FOUNDED_BY, LOCATED_IN, COLLABORATED_WITH,
                        HAS_SKILL, PUBLISHED_ON, MENTIONS, PART_OF

User:   Text: "{chunk_text}"

        Output format:
        {
          "entities": [
            {"id": "e1", "text": "...", "type": "...", "span": [start, end]}
          ],
          "relations": [
            {"head": "e1", "tail": "e2", "type": "..."}
          ]
        }
```

### Output Schema
```python
@dataclass
class Entity:
    id:     str
    text:   str
    type:   str          # PERSON, ORG, etc.
    span:   List[int]    # [start_char, end_char] in chunk
    canonical_id: Optional[str]  # set after deduplication

@dataclass
class Relation:
    head:   str          # entity id
    tail:   str          # entity id
    type:   str

@dataclass
class ChunkExtraction:
    chunk_id:  str
    chunk_text: str
    source_block_id: str
    source_page: int
    entities:  List[Entity]
    relations: List[Relation]
```

### Batching for Speed
- Batch multiple chunks into a single inference call using `transformers` batch generation
- Use `vllm` or `text-generation-inference` for high-throughput serving if processing large documents

---

## 8. Phase 6 — Fuzzy Deduplication

### Problem
Across chunks, the same real-world entity may appear as:
- `"Jon"` vs `"John"` (typing error)
- `"Micr0s0ft"` vs `"Microsoft"` (OCR noise, partially corrected)
- `"Microsoft Corp"` vs `"Microsoft Corporation"` (abbreviation)
- `"ML"` vs `"Machine Learning"` (acronym) ← separate handling needed

### Algorithm
Located in `postprocessing/deduplicator.py`

```
Input: All entities from all ChunkExtractions

Step 1 — Blocking (reduces O(n²) to O(n log n))
   Group entities by: entity_type + first 3 characters of lowercased text
   Only compare entities within the same block-key
   E.g., "Jon"→PERSON and "John"→PERSON share key "PERSON:jon"

Step 2 — Pairwise Similarity (RapidFuzz)
   from rapidfuzz import fuzz
   for each pair (a, b) in same block:
       score = fuzz.token_set_ratio(a.text, b.text)
       # token_set_ratio handles reordering: "John Smith" vs "Smith, John"
       if score >= THRESHOLD:
           mark as duplicate

Step 3 — Union-Find Merge
   Use a Union-Find (Disjoint Set Union) data structure to merge duplicate sets
   Canonical entity = most frequent text in the merged set
   (or longest text, configurable)

Step 4 — Relation Update
   For all relations: replace head/tail entity IDs with canonical_id
   Remove self-referencing relations (head canonical == tail canonical)

Step 5 — Acronym Expansion (optional, future)
   Maintain an acronym registry populated during extraction
   "ML" appears near "Machine Learning" → link them
```

### Threshold Configuration
| Scenario | Recommended Threshold |
|---|---|
| OCR noise correction (short words ≤ 6 chars) | ≥ 92 |
| Person names | ≥ 88 |
| Organisation names | ≥ 85 |
| General entities | ≥ 88 |

> **Default**: 88. Expose as a slider in the UI under "Advanced Settings".

### Libraries
- `rapidfuzz` — C++ backed, 100× faster than `python-Levenshtein`
- `networkx` — for building entity graph and connected components (alternative to manual Union-Find)

### Output Schema
```python
@dataclass
class DedupedExtractionResult:
    document_id:   str
    entities:      List[Entity]          # canonical entities only
    relations:     List[Relation]        # updated to canonical IDs
    chunk_extractions: List[ChunkExtraction]  # raw, for traceability
    dedup_log:     List[Dict]            # {"merged": ["Jon", "John"], "canonical": "John"}
    metadata:      Dict[str, Any]
```

---

## 9. Phase 7 — UI Chunking Controls (Streamlit)

### New UI Section: "🧩 Extraction Settings"
Add a collapsible sidebar panel in `streamlit_app_unified.py` shown when a PDF or image is uploaded:

```
┌──────────────────────────────────────────────┐
│  🧩 Extraction Settings                      │
├──────────────────────────────────────────────┤
│  Chunking Strategy                           │
│  ● Sentence/Line                             │
│  ○ Paragraph/Section                         │
│  ○ Page                                      │
│  ○ Context (BERTopic)  [ℹ Clusters related  │
│                          paragraphs]         │
│                                              │
│  Max Tokens per Chunk     [512  ──────]      │
│  Chunk Overlap            [64   ───]         │
│                                              │
│  Deduplication Threshold  [88%  ──────]      │
│                                              │
│  ☑ Spell Correction                          │
│  ☑ Entity Extraction                         │
│  ☑ Relation Extraction                       │
│  ☐ Export Entity Graph (GraphML)             │
└──────────────────────────────────────────────┘
```

### New Results Tabs (extended from current 3 tabs → 5 tabs)
```
📋 Extracted Content  |  🏷️ Entities & Relations  |  🧩 Chunks  |  🔍 Technical Details  |  📦 Export
```

**🏷️ Entities & Relations tab:**
- Entities table: text, type, frequency, canonical form
- Relations table: head entity, relation type, tail entity, source chunk
- Entity graph preview using `pyvis` (interactive HTML graph)

**🧩 Chunks tab:**
- List all chunks with source metadata (page, block, VLM label)
- Show which entities were found in each chunk

---

## 10. Schema Extensions

### Current → Extended Data Flow

```
DocumentObject          (existing, unchanged)
    │
    ▼
LayoutBlock[]           (NEW — from VLM)
    │
    ▼
Block (extended)        (EXTENDED — adds title, label, corrected_text)
    │ contains
    ▼
Region[]                (existing, word-level OCR output)
    │
    ▼
Chunk[]                 (NEW — from chunking engine)
    │
    ▼
ChunkExtraction[]       (NEW — from SLM)
    │
    ▼
DedupedExtractionResult (NEW — final output)
```

### Updated `binary_schema.py` summary
```python
# EXISTING (unchanged)
Region(region_id, text, bbox, confidence, metadata)

# EXTENDED
Block(block_id, title, label, bbox, regions, raw_text,
      corrected_text, confidence, metadata)

Page(page_id, page_number, blocks, metadata)
# property: regions = [r for b in blocks for r in b.regions]  ← backwards compat

BinaryDocument(document_id, pages, metadata)

# NEW
LayoutBlock(block_id, label, bbox, confidence, page_number)
Chunk(chunk_id, text, strategy, source_blocks, token_count, metadata)
Entity(id, text, type, span, canonical_id)
Relation(head, tail, type)
ChunkExtraction(chunk_id, chunk_text, source_block_id, source_page, entities, relations)
DedupedExtractionResult(document_id, entities, relations, chunk_extractions, dedup_log, metadata)
```

---

## 11. GPU / Infrastructure Plan

### Memory Budget (24 GB VRAM)

| Component | Model | VRAM |
|---|---|---|
| PaliGemma 2 3B (VLM layout) | BF16 | ~6 GB |
| neuspell BERT (spell correction) | BF16 | ~0.5 GB |
| BGE-M3 embeddings (BERTopic) | BF16 | ~2.2 GB |
| Qwen2.5-7B SLM (extraction) | BF16 | ~14 GB |
| **Total (all loaded)** | | **~22.7 GB** |
| **Available headroom** | | **~1.3 GB** |

> VLM and SLM can share GPU memory if loaded sequentially (process one page: VLM → OCR → correction → chunk → SLM, then move to next). Use `torch.cuda.empty_cache()` between VLM and SLM calls if VRAM is tight.

> **Alternatively**: Run PaliGemma 2 3B in INT8 quantisation (~3 GB) using `bitsandbytes` → frees ~3 GB extra headroom.

### HuggingFace Setup Options

**Option A — HuggingFace Inference Endpoints**
- Deploy Qwen2.5-7B fine-tuned on a dedicated endpoint (A10G 24 GB)
- `etl_pipeline` calls REST API → no local GPU needed for SLM
- Cost: ~$1.3/hr for A10G

**Option B — HuggingFace Spaces (ZeroGPU)**
- Deploy entire pipeline as a Space with `@spaces.GPU` decorator
- Free tier available; queue-based

**Option C — Local with 24 GB GPU**
- Load all models locally
- Use `accelerate` for device mapping if running on multi-GPU

### Model Loading Best Practices
```python
# Load once, reuse across documents
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",          # uses accelerate for auto device placement
    low_cpu_mem_usage=True
)
# For PaliGemma (VLM):
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
vlm = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-3b-pt-224",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

---

## 12. New File Structure

```
etl_pipeline/
├── main.py                               (existing, minor update for new pipeline)
├── requirements.txt                      (update with new deps)
│
├── handlers/
│   ├── binary_handler.py                 (UPDATE: calls VLM before OCR)
│   ├── binary_schema.py                  (UPDATE: add Block, extend Page)
│   ├── ocr/
│   │   ├── chandra_adapter.py            (UPDATE: add block-aware OCR function)
│   │   ├── dispatcher.py                 (UPDATE: pass layout blocks)
│   │   ├── image.py                      (UPDATE: VLM → crop → OCR)
│   │   └── pdf.py                        (UPDATE: rasterise + VLM + OCR)
│   └── vlm/                              ← NEW MODULE
│       ├── __init__.py
│       ├── paligemma_adapter.py          (PaliGemma model loading + inference)
│       ├── layout_detector.py            (orchestrate VLM for full doc)
│       └── block_schema.py               (LayoutBlock dataclass)
│
├── postprocessing/                       ← NEW MODULE
│   ├── __init__.py
│   ├── pipeline.py                       (orchestrates steps 3–6)
│   ├── spell_correction.py               (Stage A regex + Stage B SymSpell + Stage C neuspell)
│   ├── chunker.py                        (strategies A, B, C; dispatches to bertopic for D)
│   ├── bertopic_chunker.py               (BERTopic context-based chunking)
│   ├── slm_extractor.py                  (SLM loading + batch entity/relation extraction)
│   ├── deduplicator.py                   (RapidFuzz blocking + Union-Find merge)
│   └── extraction_schema.py              (Chunk, Entity, Relation, ChunkExtraction,
│                                          DedupedExtractionResult dataclasses)
│
├── ingestion/                            (mostly unchanged)
│   └── ...
│
└── ui/
    └── streamlit_app_unified.py          (UPDATE: chunking controls, entity/relation tabs)
```

---

## 13. New Dependencies

Add to `requirements.txt`:

```txt
# VLM — PaliGemma
transformers>=4.46.0
accelerate>=0.30.0
bitsandbytes>=0.43.0          # INT8/INT4 quantisation
pillow>=10.0.0
torch>=2.2.0

# PDF Rasterisation
pymupdf>=1.24.0               # fitz — fast PDF → image

# Spell Correction
symspellpy>=6.7.7
neuspell>=1.0.0               # BERT-based context spell check

# Embeddings + BERTopic
sentence-transformers>=3.0.0
bertopic>=0.16.4
umap-learn>=0.5.6
hdbscan>=0.8.38

# NLP Tokenisation / Sentence Splitting
spacy>=3.7.0
# python -m spacy download en_core_web_sm

# Fuzzy Deduplication
rapidfuzz>=3.6.0
networkx>=3.2.0

# Entity Graph Export
pyvis>=0.3.2

# Existing (confirm versions)
PyPDF2>=3.0.0
chardet>=5.0.0
beautifulsoup4>=4.12.0
```

---

## 14. Implementation Roadmap (Sprints)

### Sprint 1 — VLM Integration (Week 1–2)
- [ ] Create `handlers/vlm/` module
- [ ] Implement `paligemma_adapter.py` with model loading + JSON-output prompting
- [ ] Implement `layout_detector.py` with PDF rasterisation (PyMuPDF) and image crop logic
- [ ] Add `LayoutBlock` dataclass to `block_schema.py`
- [ ] Update `binary_schema.py`: extend `Page` with `blocks: List[Block]`
- [ ] Update `handlers/ocr/pdf.py` and `image.py` to receive `LayoutBlock` list and OCR per block
- [ ] Update `binary_handler.py` to call VLM before OCR
- [ ] Unit tests: mock VLM output, verify block crop + OCR routing

### Sprint 2 — Spell Correction (Week 2–3)
- [ ] Implement `postprocessing/spell_correction.py`
  - [ ] Stage A: OCR regex substitution patterns (confidence-gated)
  - [ ] Stage B: SymSpell integration with English frequency dict
  - [ ] Stage C: neuspell BERT correction for ambiguous tokens
- [ ] Add `raw_text` and `corrected_text` fields to `Block`
- [ ] Integrate into `binary_handler.py` post-OCR
- [ ] Test on deliberately degraded OCR samples from `test_cases/`

### Sprint 3 — Chunking Engine (Week 3–4)
- [ ] Implement `postprocessing/chunker.py` with strategies A (sentence), B (paragraph), C (page)
- [ ] Implement `postprocessing/bertopic_chunker.py` with full BERTopic pipeline
- [ ] Add `Chunk` dataclass to `extraction_schema.py`
- [ ] Wire chunker to `postprocessing/pipeline.py`
- [ ] Unit tests: verify each strategy splits correctly, respects `max_tokens`

### Sprint 4 — SLM Extraction (Week 4–6)
- [ ] Fine-tune Qwen2.5-7B on entity/relation extraction dataset (use existing NER datasets + custom domain data)
- [ ] Implement `postprocessing/slm_extractor.py`
  - [ ] Model loading with `device_map="auto"`, `torch.bfloat16`
  - [ ] Batch inference loop over chunks
  - [ ] JSON output parsing with retry on malformed JSON
- [ ] Add `Entity`, `Relation`, `ChunkExtraction` dataclasses
- [ ] Error handling: log failed chunks, return partial results

### Sprint 5 — Fuzzy Deduplication (Week 6–7)
- [ ] Implement `postprocessing/deduplicator.py`
  - [ ] Blocking by `entity_type + first_3_chars`
  - [ ] RapidFuzz `token_set_ratio` pairwise scoring
  - [ ] Union-Find canonical entity resolution
  - [ ] Relation head/tail ID update
- [ ] Add `DedupedExtractionResult` to `extraction_schema.py`
- [ ] Test with synthetic duplicate entities (Jon/John, corp/corporation variants)

### Sprint 6 — UI & Integration (Week 7–8)
- [ ] Update `streamlit_app_unified.py`:
  - [ ] Add "Extraction Settings" sidebar panel (strategy selector, sliders)
  - [ ] Add "🏷️ Entities & Relations" tab with table + pyvis graph
  - [ ] Add "🧩 Chunks" tab
  - [ ] Add "Export Entity Graph (GraphML)" download button
- [ ] Update `postprocessing/pipeline.py` to orchestrate all 6 steps
- [ ] End-to-end integration test with LinkedIn screenshot and a multi-page PDF
- [ ] Performance profiling: log time per step, VRAM usage

### Sprint 7 — Testing, Docs & Optimisation (Week 8–9)
- [ ] Write test cases in `test_cases/` for each new module
- [ ] Add `test_cases/sample_linkedin_screenshot.png` and `test_cases/sample_multi_page.pdf`
- [ ] Profile VRAM: ensure PaliGemma + Qwen2.5 fit in 24 GB
- [ ] Add INT8 quantisation fallback using `bitsandbytes` for memory-constrained environments
- [ ] Update `README.md` with Phase 2 architecture
- [ ] Update `etl_pipeline/ui/README.md` with new UI controls

---

## Quick Reference: Data Objects Per Stage

```
Stage          Input                        Output
─────────────────────────────────────────────────────────────────────
Ingest         raw file/bytes               DocumentObject
VLM Detection  DocumentObject (image/PDF)   List[LayoutBlock]
OCR            LayoutBlock + cropped image  List[Block] (with title)
Spell Correct  Block.raw_text               Block.corrected_text
Chunking       List[Block]                  List[Chunk]
SLM Extract    Chunk.text                   ChunkExtraction
Dedup          List[ChunkExtraction]        DedupedExtractionResult
UI Display     DedupedExtractionResult      Streamlit tables, graph, JSON
```

---

*Planner last updated: February 19, 2026*
