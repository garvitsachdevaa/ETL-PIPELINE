"""
tests/test_ocr_routing.py

Unit tests for Task 7 — VLM layout-guided OCR routing.
Covers ocr/image.py, ocr/dispatcher.py, and binary_handler.py.

NO GPU, NO Chandra CLI, NO PaliGemma model required.
All VLM and Chandra calls are mocked.
"""

import io
import uuid
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from handlers.vlm.block_schema import LayoutBlock
from handlers.binary_schema import Page, Block, Region
from handlers.ocr.dispatcher import run_ocr
from handlers.ocr.image import ocr_image
from handlers.ocr.pdf import ocr_pdf
from handlers.binary_handler import handle_binary


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_layout_block(label="header", bbox=None, page=1) -> LayoutBlock:
    return LayoutBlock(
        block_id=str(uuid.uuid4()),
        label=label,
        bbox=bbox or [0, 0, 800, 100],
        confidence=0.95,
        page_number=page,
    )


def _chandra_result(block_id, label, text="Sample text") -> dict:
    words = [{"text": w, "bbox": [0, 0, 50, 20], "conf": 0.95} for w in text.split()]
    return {
        "block_id":   block_id,
        "title":      f"{label.title()} Section",
        "label":      label,
        "text":       text,
        "words":      words,
        "confidence": 0.95,
    }


def _png_bytes(width=800, height=1000) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _make_doc(mime_type="image/png", fmt="image") -> MagicMock:
    doc = MagicMock()
    doc.mime_type = mime_type
    doc.detected_format = fmt
    doc.document_id = str(uuid.uuid4())
    doc.raw_bytes = _png_bytes()
    return doc


# ---------------------------------------------------------------------------
# ocr/image.py — layout-guided path
# ---------------------------------------------------------------------------

class TestOcrImageWithLayout:

    def test_builds_one_block_per_layout_block(self):
        lb1 = _make_layout_block("header", [0, 0, 800, 100])
        lb2 = _make_layout_block("body",   [0, 100, 800, 900])
        doc = _make_doc()

        with patch("handlers.ocr.image.run_chandra_on_block") as mock_ocr:
            mock_ocr.side_effect = lambda image, block_id, block_label: _chandra_result(block_id, block_label)
            pages = ocr_image(doc, layout_blocks=[lb1, lb2])

        assert len(pages) == 1
        assert len(pages[0].blocks) == 2
        assert pages[0].blocks[0].label == "header"
        assert pages[0].blocks[1].label == "body"
        assert mock_ocr.call_count == 2

    def test_block_bbox_comes_from_vlm(self):
        lb = _make_layout_block("table", [50, 200, 750, 500])
        doc = _make_doc()

        with patch("handlers.ocr.image.run_chandra_on_block") as mock_ocr:
            mock_ocr.return_value = _chandra_result(lb.block_id, lb.label)
            pages = ocr_image(doc, layout_blocks=[lb])

        assert pages[0].blocks[0].bbox == [50, 200, 750, 500]

    def test_block_id_matches_layout_block(self):
        lb = _make_layout_block("figure")
        doc = _make_doc()

        with patch("handlers.ocr.image.run_chandra_on_block") as mock_ocr:
            mock_ocr.return_value = _chandra_result(lb.block_id, lb.label)
            pages = ocr_image(doc, layout_blocks=[lb])

        assert pages[0].blocks[0].block_id == lb.block_id

    def test_raw_text_from_chandra(self):
        lb = _make_layout_block("body")
        doc = _make_doc()

        with patch("handlers.ocr.image.run_chandra_on_block") as mock_ocr:
            mock_ocr.return_value = _chandra_result(lb.block_id, lb.label, "Hello world")
            pages = ocr_image(doc, layout_blocks=[lb])

        assert pages[0].blocks[0].raw_text == "Hello world"

    def test_word_level_regions_built(self):
        lb = _make_layout_block("body")
        doc = _make_doc()

        with patch("handlers.ocr.image.run_chandra_on_block") as mock_ocr:
            mock_ocr.return_value = {
                "block_id": lb.block_id, "title": "", "label": lb.label,
                "text": "Hello world",
                "words": [
                    {"text": "Hello", "bbox": [0, 0, 50, 20], "conf": 0.98},
                    {"text": "world", "bbox": [55, 0, 110, 20], "conf": 0.96},
                ],
                "confidence": 0.97,
            }
            pages = ocr_image(doc, layout_blocks=[lb])

        regions = pages[0].blocks[0].regions
        assert len(regions) == 2
        assert regions[0].text == "Hello"
        assert regions[1].text == "world"

    def test_page_metadata_layout_guided_true(self):
        lb = _make_layout_block("body")
        doc = _make_doc()

        with patch("handlers.ocr.image.run_chandra_on_block") as mock_ocr:
            mock_ocr.return_value = _chandra_result(lb.block_id, lb.label)
            pages = ocr_image(doc, layout_blocks=[lb])

        assert pages[0].metadata.get("layout_guided") is True

    def test_fallback_to_whole_image_when_layout_blocks_none(self):
        doc = _make_doc()

        with patch("handlers.ocr.image.run_chandra_cli") as mock_cli:
            mock_cli.return_value = {
                "blocks": [{"text": "whole page", "bbox": [0, 0, 800, 1000], "confidence": 0.9}]
            }
            pages = ocr_image(doc, layout_blocks=None)

        mock_cli.assert_called_once()
        assert len(pages) == 1

    def test_backwards_compat_regions_property(self):
        """Page.regions returns all regions across all blocks (backwards compat)."""
        lb1 = _make_layout_block("header")
        lb2 = _make_layout_block("body")
        doc = _make_doc()

        with patch("handlers.ocr.image.run_chandra_on_block") as mock_ocr:
            mock_ocr.side_effect = lambda image, block_id, block_label: {
                "block_id": block_id, "title": "", "label": block_label,
                "text": "word",
                "words": [{"text": "word", "bbox": [0, 0, 50, 20], "conf": 0.9}],
                "confidence": 0.9,
            }
            pages = ocr_image(doc, layout_blocks=[lb1, lb2])

        # 2 blocks × 1 region each
        assert len(pages[0].regions) == 2


# ---------------------------------------------------------------------------
# ocr/dispatcher.py — routing
# ---------------------------------------------------------------------------

class TestOcrDispatcher:

    def test_image_doc_routes_to_ocr_image(self):
        doc = _make_doc("image/png", "image")
        lb = [_make_layout_block()]

        with patch("handlers.ocr.dispatcher.ocr_image", return_value=[MagicMock()]) as mock_img:
            run_ocr(doc, layout_blocks=lb)

        mock_img.assert_called_once_with(doc, layout_blocks=lb)

    def test_pdf_doc_routes_to_ocr_pdf(self):
        doc = _make_doc("application/pdf", "document")
        lb = [[_make_layout_block()]]

        with patch("handlers.ocr.dispatcher.ocr_pdf", return_value=[MagicMock()]) as mock_pdf:
            run_ocr(doc, layout_blocks=lb)

        mock_pdf.assert_called_once_with(doc, layout_blocks=lb)

    def test_no_layout_blocks_forwarded_as_none(self):
        doc = _make_doc("image/png", "image")

        with patch("handlers.ocr.dispatcher.ocr_image", return_value=[MagicMock()]) as mock_img:
            run_ocr(doc)

        mock_img.assert_called_once_with(doc, layout_blocks=None)

    def test_unsupported_format_raises(self):
        doc = _make_doc()
        doc.detected_format = "unknown"

        with pytest.raises(ValueError, match="Unsupported format"):
            run_ocr(doc)

    def test_non_pdf_document_raises(self):
        doc = _make_doc("application/msword", "document")

        with pytest.raises(ValueError):
            run_ocr(doc)


# ---------------------------------------------------------------------------
# binary_handler.py — PaliGemma → Chandra pipeline
# ---------------------------------------------------------------------------

class TestRunVlmLayout:
    """Tests for binary_handler._run_vlm_layout()."""

    def test_pdf_calls_run_layout_detection_on_pdf(self):
        doc = _make_doc("application/pdf", "document")

        with patch("handlers.binary_handler.run_layout_detection_on_pdf",
                   return_value=[[_make_layout_block()]]) as mock_vlm:
            from handlers.binary_handler import _run_vlm_layout
            result = _run_vlm_layout(doc, "pdf")

        mock_vlm.assert_called_once_with(doc.raw_bytes)
        assert result is not None

    def test_image_calls_run_layout_detection_on_image(self):
        doc = _make_doc("image/png", "image")

        with patch("handlers.binary_handler.run_layout_detection_on_image",
                   return_value=[_make_layout_block()]) as mock_vlm:
            from handlers.binary_handler import _run_vlm_layout
            result = _run_vlm_layout(doc, "image")

        assert mock_vlm.call_count == 1
        assert result is not None

    def test_returns_none_on_exception(self):
        doc = _make_doc("application/pdf", "document")

        with patch("handlers.binary_handler.run_layout_detection_on_pdf",
                   side_effect=RuntimeError("GPU OOM")):
            from handlers.binary_handler import _run_vlm_layout
            result = _run_vlm_layout(doc, "pdf")

        assert result is None


class TestHandleBinaryVlmIntegration:

    def test_vlm_layout_passed_to_run_ocr_for_pdf(self):
        """handle_binary forwards PaliGemma layout blocks to run_ocr."""
        doc = _make_doc("application/pdf", "document")
        layout_blocks = [[_make_layout_block()]]

        with patch("handlers.binary_handler._run_vlm_layout",
                   return_value=layout_blocks) as _mock_vlm, \
             patch("handlers.binary_handler.run_ocr", return_value=[]) as mock_ocr:
            handle_binary(doc)

        _, kwargs = mock_ocr.call_args
        assert kwargs["layout_blocks"] is layout_blocks

    def test_vlm_layout_passed_to_run_ocr_for_image(self):
        doc = _make_doc("image/png", "image")
        layout_blocks = [_make_layout_block()]

        with patch("handlers.binary_handler._run_vlm_layout",
                   return_value=layout_blocks), \
             patch("handlers.binary_handler.run_ocr", return_value=[]) as mock_ocr:
            handle_binary(doc)

        _, kwargs = mock_ocr.call_args
        assert kwargs["layout_blocks"] is layout_blocks

    def test_run_ocr_gets_none_when_vlm_fails(self):
        """If _run_vlm_layout returns None, run_ocr is called with layout_blocks=None."""
        doc = _make_doc("image/png", "image")

        with patch("handlers.binary_handler._run_vlm_layout", return_value=None), \
             patch("handlers.binary_handler.run_ocr", return_value=[]) as mock_ocr:
            handle_binary(doc)

        _, kwargs = mock_ocr.call_args
        assert kwargs["layout_blocks"] is None

    def test_metadata_vlm_layout_guided_true_when_vlm_succeeds(self):
        doc = _make_doc("image/png", "image")

        with patch("handlers.binary_handler._run_vlm_layout",
                   return_value=[_make_layout_block()]), \
             patch("handlers.binary_handler.run_ocr", return_value=[]):
            result = handle_binary(doc)

        assert result.metadata["vlm_layout_guided"] is True
        assert result.metadata["extraction_method"] == "vlm_layout_ocr"

    def test_metadata_vlm_layout_guided_false_when_vlm_fails(self):
        doc = _make_doc("image/png", "image")

        with patch("handlers.binary_handler._run_vlm_layout", return_value=None), \
             patch("handlers.binary_handler.run_ocr", return_value=[]):
            result = handle_binary(doc)

        assert result.metadata["vlm_layout_guided"] is False
        assert result.metadata["extraction_method"] == "chandra_ocr"

    def test_unsupported_mime_returns_empty_doc(self):
        doc = _make_doc("application/zip", "archive")

        result = handle_binary(doc)

        assert result.pages == []
        assert "error" in result.metadata
