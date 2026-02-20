"""
handlers/vlm — VLM Layout Detection package (Phase 1)

Public surface:
    LayoutBlock                    — dataclass for a single detected block
    run_layout_detection_on_image  — detect blocks on a PIL image
    run_layout_detection_on_pdf    — detect blocks on every page of a PDF
    crop_image_to_block            — crop a PIL image to a LayoutBlock bbox
    unload_model                   — free PaliGemma from GPU memory
"""

from handlers.vlm.block_schema import LayoutBlock
from handlers.vlm.layout_detector import (
    run_layout_detection_on_image,
    run_layout_detection_on_pdf,
    crop_image_to_block,
)
from handlers.vlm.paligemma_adapter import unload_model

__all__ = [
    "LayoutBlock",
    "run_layout_detection_on_image",
    "run_layout_detection_on_pdf",
    "crop_image_to_block",
    "unload_model",
]
