"""
Image OCR — Phase 2 (VLM layout-guided).

Flow WITH layout_blocks:
  1. Load full image from raw_bytes
  2. Crop to each LayoutBlock bbox
  3. Run Chandra OCR on the cropped image
  4. Build Block objects (label from VLM, title/text from Chandra)

Flow WITHOUT layout_blocks (backwards-compatible):
  Runs run_chandra_cli on the full image bytes (legacy path).
"""

import io
import logging
import uuid
from typing import List, Optional

from handlers.binary_schema import Page, Region, Block
from handlers.ocr.chandra_adapter import run_chandra_cli, run_chandra_on_block

logger = logging.getLogger(__name__)


def ocr_image(doc, layout_blocks: Optional[List] = None) -> List[Page]:
    """
    OCR an image document.

    Args:
        doc:           DocumentObject with raw_bytes and document_id.
        layout_blocks: List[LayoutBlock] from run_layout_detection_on_image().
                       Pass None to fall back to whole-image OCR.
    Returns:
        List[Page] (always a single-element list for images)
    """
    if layout_blocks is not None:
        return _ocr_image_with_layout(doc, layout_blocks)
    return _ocr_image_whole(doc)


def _ocr_image_with_layout(doc, layout_blocks: List) -> List[Page]:
    """Block-aware OCR: crop to each LayoutBlock and run Chandra per crop."""
    from PIL import Image
    from handlers.vlm.layout_detector import crop_image_to_block

    img = Image.open(io.BytesIO(doc.raw_bytes))
    blocks = []

    for layout_block in layout_blocks:
        cropped = crop_image_to_block(img, layout_block)
        ocr_result = run_chandra_on_block(
            image=cropped,
            block_id=layout_block.block_id,
            block_label=layout_block.label,
        )

        regions = [
            Region(
                region_id=str(uuid.uuid4()),
                text=w.get("text", ""),
                bbox=w.get("bbox", [0, 0, 0, 0]),
                confidence=w.get("conf", 1.0),
                metadata={"engine": "chandra"},
            )
            for w in ocr_result.get("words", [])
        ]

        blocks.append(Block(
            block_id=layout_block.block_id,
            title=ocr_result.get("title", ""),
            label=layout_block.label,
            bbox=layout_block.bbox,
            regions=regions,
            raw_text=ocr_result.get("text", ""),
            confidence=ocr_result.get("confidence", 1.0),
            metadata={
                "engine": "chandra",
                "vlm_label": layout_block.label,
            },
        ))

    return [Page(
        page_id=str(uuid.uuid4()),
        page_number=1,
        blocks=blocks,
        metadata={"image": True, "layout_guided": True},
    )]


def _ocr_image_whole(doc) -> List[Page]:
    """Legacy whole-image OCR — used when no layout blocks are provided."""
    ocr_data = run_chandra_cli(doc.raw_bytes, doc.document_id)
    blocks = []

    for raw_block in ocr_data.get("blocks", []):
        region = Region(
            region_id=str(uuid.uuid4()),
            text=raw_block.get("text", ""),
            bbox=raw_block.get("bbox", [0, 0, 0, 0]),
            confidence=raw_block.get("confidence", 1.0),
            metadata={"engine": "chandra"},
        )
        blocks.append(Block(
            block_id=str(uuid.uuid4()),
            title=raw_block.get("title", ""),
            label=raw_block.get("label", "unknown"),
            bbox=raw_block.get("bbox", [0, 0, 0, 0]),
            regions=[region],
            raw_text=raw_block.get("text", ""),
            confidence=raw_block.get("confidence", 1.0),
            metadata={"engine": "chandra"},
        ))

    return [Page(
        page_id=str(uuid.uuid4()),
        page_number=1,
        blocks=blocks,
        metadata={"image": True},
    )]
