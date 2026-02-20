"""
PDF OCR — Phase 2 (VLM layout-guided).

Flow WITH layout_blocks:
  1. Rasterise each PDF page to a PIL image via PyMuPDF (150 DPI)
  2. Crop to each LayoutBlock bbox
  3. Run Chandra OCR on the cropped image
  4. Build Block objects (label from VLM, title/text from Chandra)

Flow WITHOUT layout_blocks (backwards-compatible):
  Runs run_chandra_cli on the full PDF bytes (legacy path).
"""

import io
import logging
import uuid
from typing import List, Optional

from handlers.binary_schema import Page, Region, Block
from handlers.ocr.chandra_adapter import run_chandra_cli, run_chandra_on_block

logger = logging.getLogger(__name__)

DPI = 150  # matches layout_detector.py rasterisation DPI


def ocr_pdf(doc, layout_blocks: Optional[List[List]] = None) -> List[Page]:
    """
    OCR a PDF document.

    Args:
        doc:           DocumentObject with raw_bytes and document_id.
        layout_blocks: List[List[LayoutBlock]] — one inner list per page,
                       as returned by run_layout_detection_on_pdf().
                       Pass None to fall back to whole-page OCR.
    Returns:
        List[Page]
    """
    if layout_blocks is not None:
        return _ocr_pdf_with_layout(doc, layout_blocks)
    return _ocr_pdf_whole_page(doc)


def _ocr_pdf_with_layout(doc, layout_blocks: List[List]) -> List[Page]:
    """Block-aware OCR: crop to each LayoutBlock and run Chandra per crop."""
    import fitz
    from PIL import Image
    from handlers.vlm.layout_detector import crop_image_to_block

    pages = []

    try:
        pdf = fitz.open(stream=doc.raw_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning(f"Could not open PDF for layout-guided OCR ({exc}); falling back.")
        return _ocr_pdf_whole_page(doc)

    try:
        for page_idx, (fitz_page, page_blocks) in enumerate(
            zip(pdf, layout_blocks), start=1
        ):
            blocks = []

            try:
                pix = fitz_page.get_pixmap(dpi=DPI)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
            except Exception as exc:
                logger.warning(f"Rasterisation failed for page {page_idx}: {exc}")
                pages.append(Page(
                    page_id=str(uuid.uuid4()),
                    page_number=page_idx,
                    blocks=[],
                    metadata={"pdf": True, "error": str(exc)},
                ))
                continue

            for layout_block in page_blocks:
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
                        metadata={"engine": "chandra", "page": page_idx},
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
                        "page": page_idx,
                        "vlm_label": layout_block.label,
                    },
                ))

            pages.append(Page(
                page_id=str(uuid.uuid4()),
                page_number=page_idx,
                blocks=blocks,
                metadata={"pdf": True, "layout_guided": True},
            ))
    finally:
        pdf.close()

    return pages


def _ocr_pdf_whole_page(doc) -> List[Page]:
    """Legacy whole-page OCR — used when no layout blocks are provided."""
    ocr_data = run_chandra_cli(doc.raw_bytes, doc.document_id)
    pages = []

    for page_idx, page in enumerate(ocr_data.get("pages", []), start=1):
        blocks = []
        for raw_block in page.get("blocks", []):
            region = Region(
                region_id=str(uuid.uuid4()),
                text=raw_block.get("text", ""),
                bbox=raw_block.get("bbox", [0, 0, 0, 0]),
                confidence=raw_block.get("confidence", 1.0),
                metadata={"engine": "chandra", "page": page_idx},
            )
            blocks.append(Block(
                block_id=str(uuid.uuid4()),
                title=raw_block.get("title", ""),
                label=raw_block.get("label", "unknown"),
                bbox=raw_block.get("bbox", [0, 0, 0, 0]),
                regions=[region],
                raw_text=raw_block.get("text", ""),
                confidence=raw_block.get("confidence", 1.0),
                metadata={"engine": "chandra", "page": page_idx},
            ))
        pages.append(Page(
            page_id=str(uuid.uuid4()),
            page_number=page_idx,
            blocks=blocks,
            metadata={"pdf": True},
        ))

    return pages
