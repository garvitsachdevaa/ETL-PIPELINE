from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Region:
    """Word- or line-level OCR region (unchanged from Phase 1)."""
    region_id: str
    text: str
    bbox: List[int]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class Block:
    """
    A semantically labelled content block on a page (Phase 2).

    block_id        — uuid4
    title           — Chandra-assigned block title (e.g. "User Profile Section")
    label           — VLM semantic label  (e.g. "table", "header", "user_profile")
    bbox            — [x0, y0, x1, y1] pixel coordinates from the VLM
    regions         — word/line-level OCR Region objects inside this block
    raw_text        — OCR raw text (concatenated from regions)
    corrected_text  — After OCR spell correction (Phase 3)
    confidence      — Mean OCR confidence across regions
    metadata        — Arbitrary extra data (vlm_raw, engine, etc.)
    """
    block_id: str
    title: str
    label: str
    bbox: List[int]
    regions: List[Region] = field(default_factory=list)
    raw_text: str = ""
    corrected_text: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "title": self.title,
            "label": self.label,
            "bbox": self.bbox,
            "raw_text": self.raw_text,
            "corrected_text": self.corrected_text,
            "confidence": self.confidence,
            "regions": [r.to_dict() for r in self.regions],
            "metadata": self.metadata,
        }


@dataclass
class Page:
    """
    A single page of a binary document.

    Primary field is `blocks` (List[Block]) — each block contains its own
    Region list.  The `regions` property provides a flat list of all regions
    across all blocks for backwards compatibility with existing callers.
    """
    page_id: str
    page_number: int
    blocks: List[Block] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Backwards-compatible flat regions accessor
    # ------------------------------------------------------------------
    @property
    def regions(self) -> List[Region]:
        """Flat list of all Region objects across every block on this page."""
        return [region for block in self.blocks for region in block.regions]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_number": self.page_number,
            "blocks": [block.to_dict() for block in self.blocks],
            "metadata": self.metadata,
        }


@dataclass
class BinaryDocument:
    """Top-level document object produced by binary_handler.py."""
    document_id: str
    pages: List[Page]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pages": [page.to_dict() for page in self.pages],
            "metadata": self.metadata,
        }
