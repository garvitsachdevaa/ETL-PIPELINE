from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class TextSection:
    section_id: str
    format_type: str
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'section_id': self.section_id,
            'format_type': self.format_type,
            'content': self.content,
            'metadata': self.metadata
        }

@dataclass
class TextDocument:
    document_id: str
    language: Optional[str]
    sections: List[TextSection]
    raw_text: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'language': self.language,
            'sections': [section.to_dict() for section in self.sections],
            'raw_text': self.raw_text,
            'metadata': self.metadata
        }
