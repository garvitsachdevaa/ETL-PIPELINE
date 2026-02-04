from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Region:
    region_id: str
    text: str
    bbox: List[int]
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'region_id': self.region_id,
            'text': self.text,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'metadata': self.metadata
        }

@dataclass
class Page:
    page_id: str
    page_number: int
    regions: List[Region]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'page_id': self.page_id,
            'page_number': self.page_number,
            'regions': [region.to_dict() for region in self.regions],
            'metadata': self.metadata
        }

@dataclass
class BinaryDocument:
    document_id: str
    pages: List[Page]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'pages': [page.to_dict() for page in self.pages],
            'metadata': self.metadata
        }
