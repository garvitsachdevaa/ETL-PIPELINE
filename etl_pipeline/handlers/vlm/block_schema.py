from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class LayoutBlock:
    """
    Output of the VLM layout detection step.
    Represents a single visually distinct content block on a page.
    """
    block_id: str            # uuid4
    label: str               # VLM-assigned semantic label e.g. "table", "header", "user_profile"
    bbox: List[int]          # [x0, y0, x1, y1] pixel coordinates
    confidence: float        # VLM logit confidence (1.0 if unavailable)
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "label": self.label,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }
