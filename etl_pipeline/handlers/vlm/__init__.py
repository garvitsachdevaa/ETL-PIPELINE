"""
handlers/vlm — Document Layout Detection package

Layout engine: DocLayout-YOLO (juliozhao/DocLayout-YOLO-DocStructBench)
  - ~0.5 GB VRAM, trained on DocLayNet 80k+ documents
  - Used for scanned PDFs and images

Text PDFs are handled separately by Marker (see handlers/marker_handler.py).

PaliGemma is kept in paligemma_adapter.py for VQA use only.

Public surface:
    LayoutBlock                    — dataclass for a single detected block
    run_layout_detection_on_image  — detect blocks on a PIL image
    run_layout_detection_on_pdf    — detect blocks on every page of a PDF
    crop_image_to_block            — crop a PIL image to a LayoutBlock bbox
    unload_model                   — free DocLayout-YOLO from GPU memory
"""

from handlers.vlm.block_schema import LayoutBlock
from handlers.vlm.layout_detector import (
    run_layout_detection_on_image,
    run_layout_detection_on_pdf,
    crop_image_to_block,
)
from handlers.vlm.doclayout_yolo_adapter import unload_yolo_model as unload_model

__all__ = [
    "LayoutBlock",
    "run_layout_detection_on_image",
    "run_layout_detection_on_pdf",
    "crop_image_to_block",
    "unload_model",
]
