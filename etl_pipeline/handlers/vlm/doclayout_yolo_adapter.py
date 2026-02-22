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

        # Sort top-to-bottom, left-to-right (reading order)
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

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
