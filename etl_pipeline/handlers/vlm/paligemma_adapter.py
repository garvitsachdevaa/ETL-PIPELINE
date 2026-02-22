"""
PaliGemma 2 adapter for VLM-based document layout detection.

Loads the PaliGemma model as a singleton, runs inference on a PIL image,
and parses the JSON array of layout blocks from the model output.

Model choice (from planner):
  - google/paligemma2-3b-pt-224   ~6 GB VRAM  — fast, good for dense docs
  - google/paligemma2-10b-pt-448  ~20 GB VRAM — higher resolution, better spatial precision

With 24 GB VRAM: use 10B @ 448px for single images;
quantise to INT8 (~10 GB) for batch PDF processing alongside SLM.
"""

import json
import logging
import re
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — override via detect_layout_blocks(model_id=...) if needed
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "google/paligemma2-3b-pt-224"

# PaliGemma 2 requires the prompt to START with one <image> token per image.
# Without it the processor warns and infers the token position, but the model
# output is unreliable (often plain text rather than JSON).
LAYOUT_PROMPT = (
    "<image> "
    "Identify all visually distinct content blocks in this document page. "
    "For each block return a JSON object: "
    '{"label": "<semantic_label>", "bbox": [x0, y0, x1, y1]} '
    "where bbox values are pixel coordinates of the block. "
    "Possible labels: header, body, heading, table, figure, footer, caption. "
    "Return ONLY a JSON array of all blocks, no other text."
)

# ---------------------------------------------------------------------------
# Module-level singletons (avoid re-loading on every call)
# ---------------------------------------------------------------------------
_model = None
_processor = None
_loaded_model_id: Optional[str] = None


def _load_model(model_id: str = DEFAULT_MODEL_ID) -> Tuple:
    """
    Load PaliGemma model + processor as a singleton.
    Re-loads only if model_id changes.
    """
    global _model, _processor, _loaded_model_id

    if _model is not None and _loaded_model_id == model_id:
        return _model, _processor

    try:
        import os
        import torch

        # Ensure HF authentication is active before downloading gated weights.
        # On HF Spaces the token is injected as HF_TOKEN env var.
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login as _hf_login
                _hf_login(token=hf_token, add_to_git_credential=False)
                logger.info("HF login successful for gated model access.")
            except Exception as login_exc:
                logger.warning(f"HF login failed ({login_exc}); will try token= kwarg.")

        logger.info(f"Loading PaliGemma model: {model_id}")

        from transformers import (
            PaliGemmaForConditionalGeneration,
            PaliGemmaProcessor,
            SiglipImageProcessor,
            PreTrainedTokenizerFast,
        )
        from huggingface_hub import snapshot_download

        # ── Download repo snapshot, explicitly blocking video preprocessor ──
        # PaliGemmaForConditionalGeneration.from_pretrained(model_id) in
        # transformers >=4.46 internally fetches video_preprocessor_config.json
        # and tries to instantiate VideoImageProcessor — hangs indefinitely.
        # Fix: use snapshot_download with ignore_patterns to get a clean local
        # copy WITHOUT the video preprocessor file, then load from local path.
        # local_files_only=True on from_pretrained means no network calls at
        # all after this — cannot fetch video_preprocessor_config.json.
        logger.info(">>> Downloading model snapshot (~6 GB, video preprocessor excluded)...")
        import transformers as _transformers
        _transformers.logging.set_verbosity_info()
        _transformers.logging.enable_progress_bar()

        local_dir = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            ignore_patterns=[
                "*video*",
                "video_preprocessor_config.json",
                "*.msgpack",
                "flax_model*",
                "tf_model*",
                "rust_model*",
            ],
        )
        logger.info(f"Snapshot ready at: {local_dir}")
        _transformers.logging.set_verbosity_error()

        # ── Image processor ────────────────────────────────────────────────
        logger.info("Building SiglipImageProcessor (hardcoded params)...")
        _image_processor = SiglipImageProcessor(
            do_resize=True,
            size={"height": 224, "width": 224},
            resample=3,
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        _image_processor.image_seq_length = 256  # (224/14)^2 = 256 patches

        # ── Tokenizer ──────────────────────────────────────────────────────
        import os as _os
        _tokenizer_path = _os.path.join(local_dir, "tokenizer.json")
        logger.info(f"Loading tokenizer from local snapshot: {_tokenizer_path}")
        _tokenizer = PreTrainedTokenizerFast(tokenizer_file=_tokenizer_path)

        # ── Assemble processor ─────────────────────────────────────────────
        logger.info("Assembling PaliGemmaProcessor...")
        _processor = PaliGemmaProcessor(
            image_processor=_image_processor,
            tokenizer=_tokenizer,
        )

        # ── Model weights — load from local snapshot, no network calls ─────
        logger.info("Loading PaliGemma weights from local snapshot...")
        _model = PaliGemmaForConditionalGeneration.from_pretrained(
            local_dir,                   # local path, not repo id
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            local_files_only=True,       # never fetches anything from network
        )

        _model.eval()
        _loaded_model_id = model_id
        logger.info(f"PaliGemma loaded on device: {next(_model.parameters()).device}")
        return _model, _processor

    except Exception as exc:
        logger.error(f"Failed to load PaliGemma ({model_id}): {exc}")
        raise


def detect_layout_blocks(
    image,                             # PIL.Image.Image
    page_number: int = 1,
    model_id: str = DEFAULT_MODEL_ID,
) -> List[dict]:
    """
    Run PaliGemma on a PIL image and return raw block dicts.

    Returns:
        List of {"label": str, "bbox": [x0, y0, x1, y1]} dicts.
        Falls back to a single whole-page block on any failure.
    """
    try:
        import torch

        model, processor = _load_model(model_id)

        inputs = processor(
            text=LAYOUT_PROMPT,
            images=image,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        raw_output = processor.decode(output_ids[0], skip_special_tokens=True)
        logger.debug(f"PaliGemma raw output (page {page_number}): {raw_output[:300]}")

        blocks = _parse_vlm_output(raw_output)

        if not blocks:
            logger.warning(
                f"No blocks detected on page {page_number}; using whole-page fallback."
            )
            return _whole_page_fallback(image)

        return blocks

    except Exception as exc:
        logger.error(f"VLM layout detection failed on page {page_number}: {exc}")
        return _whole_page_fallback(image)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_vlm_output(raw_output: str) -> List[dict]:
    """
    Extract a valid JSON array of block dicts from the VLM output string.

    Strategy:
      1. Try direct JSON parse of the full string.
      2. Find the first [...] substring and parse that.
      3. Validate that each entry has "label" (str) and "bbox" ([int×4]).
    """
    # 1. Direct parse
    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, list):
            return _validate_blocks(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Extract outermost JSON array in the output (greedy — spans nested brackets)
    match = re.search(r'\[.*\]', raw_output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return _validate_blocks(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Could not parse a JSON array from VLM output.")
    return []


def _validate_blocks(blocks: list) -> List[dict]:
    """Keep only entries that have a string label and a 4-int bbox."""
    valid = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        label = b.get("label")
        bbox = b.get("bbox")
        if (
            isinstance(label, str)
            and isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)
        ):
            # Normalise bbox values to int
            b["bbox"] = [int(v) for v in bbox]
            valid.append(b)
    return valid


def _whole_page_fallback(image) -> List[dict]:
    """Return a single block covering the entire image."""
    w, h = image.size
    return [{"label": "full_page", "bbox": [0, 0, w, h]}]


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def unload_model() -> None:
    """
    Release the PaliGemma model from GPU/CPU memory.
    Call this after batch VLM inference to free VRAM before loading the SLM.
    """
    global _model, _processor, _loaded_model_id
    if _model is None:
        return

    try:
        import torch
        del _model
        del _processor
        _model = None
        _processor = None
        _loaded_model_id = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("PaliGemma model unloaded and VRAM cleared.")
    except Exception as exc:
        logger.warning(f"Error during model unload: {exc}")
