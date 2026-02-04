import uuid
from handlers.binary_schema import Page, Region
from handlers.ocr.chandra_adapter import run_chandra_cli

def ocr_image(doc):
    ocr_data = run_chandra_cli(doc.raw_bytes, doc.document_id)

    regions = []

    for block in ocr_data.get("blocks", []):
        regions.append(
            Region(
                region_id=str(uuid.uuid4()),
                text=block.get("text", ""),
                bbox=block.get("bbox", [0, 0, 0, 0]),
                confidence=block.get("confidence", 1.0),
                metadata={
                    "engine": "chandra"
                }
            )
        )

    return [
        Page(
            page_id=str(uuid.uuid4()),
            page_number=1,
            regions=regions,
            metadata={"image": True}
        )
    ]
