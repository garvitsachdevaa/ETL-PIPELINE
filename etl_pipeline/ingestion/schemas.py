from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class DocumentObject:
    document_id: str
    source_name: str
    raw_bytes: Optional[bytes]
    raw_text: Optional[str]
    detected_format: str
    mime_type: Optional[str]
    encoding: Optional[str]
    language: Optional[str]
    metadata: Dict[str, Any]
    routing_target: str
