"""
DocLayout-YOLO adapter for document layout detection.

Model: juliozhao/DocLayout-YOLO-DocStructBench
  - ~0.5 GB VRAM — tiny compared to any VLM
  - Trained on DocLayNet (80 k+ real documents: reports, manuals, forms,
    patents, books, financial filings, scientific papers)
  - Detects 10 classes: title, plain text, abandon, figure,
    figure_caption, table, table_caption, table_footnote,
    isolate_formula, formula_caption
  - 'abandon' = page-noise (headers, footers, watermarks, page numbers)
    we silently skip these blocks

Output per block:
    {"label": str, "bbox": [x0, y0, x1, y1], "confidence": float}
    — coordinates are in ORIGINAL IMAGE pixel space (no normalisation needed)
    — YOLO always outputs pixel coords, unlike PaliGemma's 0-1000 space
"""

import logging
import os
import uuid
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
MODEL_FILENAME   = "doclayout_yolo_docstructbench_imgsz1024.pt"
CONF_THRESHOLD   = 0.2   # discard weak detections
IMGSZ            = 1024  # inference resolution (model was trained at 1024)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_model = None
_loaded_model_id: Optional[str] = None


def _load_model(model_id: str = DEFAULT_MODEL_ID):
    global _model, _loaded_model_id
    if _model is not None and _loaded_model_id == model_id:
        return _model

    try:
        import torch
        from huggingface_hub import hf_hub_download
        from doclayout_yolo import YOLOv10

        hf_token = os.environ.get("HF_TOKEN")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Downloading DocLayout-YOLO weights from {model_id}...")
        weights_path = hf_hub_download(
            repo_id=model_id,
            filename=MODEL_FILENAME,
            token=hf_token,
        )
        logger.info(f"DocLayout-YOLO weights at: {weights_path}")

        _model = YOLOv10(weights_path)
        _loaded_model_id = model_id
        logger.info(f"DocLayout-YOLO ready on {device}.")
        return _model

    except Exception as exc:
        logger.error(f"Failed to load DocLayout-YOLO: {exc}")
        raise


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def detect_layout_blocks_yolo(
    image,                           # PIL.Image.Image
    page_number: int = 1,
    model_id: str = DEFAULT_MODEL_ID,
) -> List[dict]:
    """
    Run DocLayout-YOLO on a PIL image.

    Returns:
        List of {"label": str, "bbox": [x0, y0, x1, y1], "confidence": float}.
        Bboxes are in original image pixel coordinates.
        Falls back to a single whole-page block on any error.
    """
    try:
        import torch

        model  = _load_model(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        results = model.predict(
            image,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            device=device,
            verbose=False,
        )

        blocks: List[dict] = []
        if results and len(results) > 0:
            result = results[0]
            names  = result.names  # {0: 'title', 1: 'plain text', 2: 'abandon', ...}

            for box in result.boxes:
                label_idx = int(box.cls.item())
                label     = names.get(label_idx, "unknown")

                # 'abandon' = noise (running headers, footers, watermarks) — skip
                if label == "abandon":
                    continue

                x0, y0, x1, y1 = box.xyxy[0].tolist()
                confidence = float(box.conf.item())

                # Normalise YOLO label names to our canonical set
                label = _normalise_label(label)

                blocks.append({
                    "label":      label,
                    "bbox":       [int(x0), int(y0), int(x1), int(y1)],
                    "confidence": confidence,
                })

        if not blocks:
            logger.warning(
                f"DocLayout-YOLO found no blocks on page {page_number}; "
                "using whole-page fallback."
            )
            w, h = image.size
            return [{"label": "full_page", "bbox": [0, 0, w, h], "confidence": 0.0}]

        # Column-aware reading order:
        #   1. Cluster blocks into vertical column bands by x-centre
        #   2. Sort columns left → right
        #   3. Within each column sort blocks top → bottom
        # This prevents interleaving of adjacent columns (e.g. LinkedIn
        # left sidebar, centre feed, right panel all mixed by raw y-sort).
        image_width = image.size[0]
        blocks = _column_aware_sort(blocks, image_width)

        logger.info(
            f"Page {page_number}: YOLO detected {len(blocks)} block(s) → "
            f"{[b['label'] for b in blocks]}"
        )
        return blocks

    except Exception as exc:
        logger.error(f"DocLayout-YOLO detection failed on page {page_number}: {exc}")
        w, h = image.size
        return [{"label": "full_page", "bbox": [0, 0, w, h], "confidence": 0.0}]


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def unload_yolo_model() -> None:
    """
    Release DocLayout-YOLO from memory.
    YOLO is only ~0.5 GB so this is usually not needed, but exposed for
    completeness / testing.
    """
    global _model, _loaded_model_id
    if _model is None:
        return
    try:
        import torch
        del _model
        _model           = None
        _loaded_model_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("DocLayout-YOLO unloaded.")
    except Exception as exc:
        logger.warning(f"Error unloading DocLayout-YOLO: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _column_aware_sort(blocks: List[dict], image_width: int) -> List[dict]:
    """
    Sort blocks in natural reading order for multi-column layouts.

    Algorithm:
      1. Compute the x-centre of every block.
      2. Use a simple 1-D gap scan on sorted x-centres to find column
         boundaries: wherever the gap between consecutive x-centres exceeds
         `col_gap_ratio * image_width` we declare a new column band.
      3. Assign every block to its column band.
      4. Final sort key: (column_index, y0) — left column first,
         top-to-bottom within each column.

    Works for 1, 2, 3 … N column layouts without needing to know N in
    advance. Falls back gracefully to pure y-sort for single-column pages.

    Args:
        blocks:      List of block dicts with 'bbox' keys.
        image_width: Pixel width of the source image (used to scale gap threshold).

    Returns:
        Reordered blocks list.
    """
    if len(blocks) <= 1:
        return blocks

    # Fraction of image width that counts as a column gap.
    # 0.08 = 8 % — catches most multi-column layouts without false-splitting
    # single-column text where blocks naturally have slight x variation.
    COL_GAP_RATIO = 0.08
    gap_threshold = image_width * COL_GAP_RATIO

    # Compute x-centre for each block
    x_centres = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in blocks]

    # Sort blocks by x-centre to find column band boundaries
    sorted_by_x = sorted(zip(x_centres, range(len(blocks))), key=lambda t: t[0])

    # Scan for gaps → column band boundaries
    # col_starts[i] = x-centre value where column i begins
    col_starts = [sorted_by_x[0][0]]
    for i in range(1, len(sorted_by_x)):
        prev_x = sorted_by_x[i - 1][0]
        curr_x = sorted_by_x[i][0]
        if curr_x - prev_x > gap_threshold:
            col_starts.append(curr_x)

    def _assign_column(x_centre: float) -> int:
        """Return 0-based column index for a given x-centre."""
        col = 0
        for i, start in enumerate(col_starts):
            # Use midpoint between adjacent column starts as the boundary
            if i + 1 < len(col_starts):
                boundary = (col_starts[i] + col_starts[i + 1]) / 2
                if x_centre < boundary:
                    return i
            else:
                return i
        return col

    # Assign column index to every block
    for b, xc in zip(blocks, x_centres):
        b["_col"] = _assign_column(xc)

    # Final sort: column first, then y0 within column
    blocks.sort(key=lambda b: (b["_col"], b["bbox"][1]))

    # Clean up temporary key
    for b in blocks:
        b.pop("_col", None)

    return blocks


def _normalise_label(yolo_label: str) -> str:
    """
    Map DocLayout-YOLO class names to our canonical block label vocabulary.
    YOLO uses 'plain text'; our schema uses 'body'.
    """
    _MAP = {
        "plain text":       "body",
        "figure_caption":   "caption",
        "table_caption":    "caption",
        "table_footnote":   "footer",
        "isolate_formula":  "formula",
        "formula_caption":  "caption",
    }
    return _MAP.get(yolo_label, yolo_label)
