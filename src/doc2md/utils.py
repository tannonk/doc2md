import argparse
import base64
import hashlib
import io
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml
from loguru import logger
from PIL import Image


class _QuotedStr(str):
    """Marker subtype forcing double-quoted YAML output for one field's
    value, independent of the emitter's automatic scalar-style selection."""


def _represent_quoted_str(dumper: yaml.Dumper, data: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


yaml.SafeDumper.add_representer(_QuotedStr, _represent_quoted_str)


def serialize_args(args: argparse.Namespace) -> str:
    """Serialize command-line arguments to a JSON string.
    Path objects are converted to strings.

    Args:
        args: Parsed command-line arguments.
    """
    args_dict = vars(args).copy()
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
    return json.dumps(args_dict, indent=2, sort_keys=True, ensure_ascii=False)


def list_served_models(base_url: str) -> list[str]:
    """Return the model ids a vLLM (or other OpenAI-compatible) server reports.

    Args:
        base_url: The URL of the server to query.

    Raises:
        RuntimeError: If the server cannot be reached.
    """
    try:
        response = requests.get(urljoin(base_url, "v1/models"), timeout=2)
        response.raise_for_status()
        return [m.get("id") for m in response.json().get("data", [])]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cannot reach vLLM server at {base_url}: {e}") from e


def check_model_available(base_url: str, model: str) -> None:
    """
    Check if the specified model is available on the vLLM server.
    Raises a RuntimeError if the model is not listed in the server's available models.
    Args:
        base_url: The URL of the vLLM server to check.
        model: The model identifier to look for in the server's available models.
    """
    models = list_served_models(base_url)
    if model not in models:
        raise RuntimeError(
            f"Model '{model}' not found on vLLM server at {base_url}. Available models: {models}"
        )


FAILURE_COUNTS: Counter[tuple[str, str]] = Counter()


def record_failure(stage: str, exc: BaseException) -> str:
    """Tally a swallowed batch-guard exception by (stage, exception type).

    The batch scripts deliberately catch ``Exception`` so one bad item cannot
    kill a long run. Tallying the types lets us see which errors really occur
    (see ``log_failure_summary``) and later narrow each ``except`` to them.

    Args:
        stage: Short name of the failing step, e.g. ``"crop"`` or ``"ocr"``.
        exc: The caught exception.

    Returns:
        ``"<ExcType>: <message>"`` for use in log lines and placeholders.
    """
    exc_type = type(exc).__qualname__
    FAILURE_COUNTS[(stage, exc_type)] += 1
    return f"{exc_type}: {exc}"


def log_failure_summary() -> None:
    """Log one line per (stage, exception type) recorded by ``record_failure``."""
    if not FAILURE_COUNTS:
        return
    logger.info("Swallowed exception summary (stage, type, count):")
    for (stage, exc_type), count in sorted(FAILURE_COUNTS.items()):
        logger.info(f"  {stage:<14} {exc_type:<40} {count}")


def compute_file_hash(path: Path, length: int = 12) -> str:
    """Short, stable content hash identifying an input file throughout this
    pipeline — used in place of its own (possibly long or special-character)
    name for the output Markdown filename and its image-cache subdirectory.
    Deterministic across runs: an unchanged file always hashes the same way,
    so re-running the script still hits the --force/skip-if-exists check.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def build_front_matter(
    *,
    file_hash: str,
    raw_file_path: str,
    conversion_method: str,
    extractor: str,
    ocr_failed_pages: list[int] | None = None,
) -> str:
    """YAML front matter recording file identity and processing provenance,
    prepended to every converted Markdown output for traceability back to the
    original (hash-renamed) source file.

    Args:
        extractor: What produced the Markdown, the same field for every
            method — the VLM model id (e.g. ``"lightonai/LightOnOCR-2-1B"``)
            or the native library and version (e.g. ``"anydoc v0.2.4"``).
        ocr_failed_pages: 1-indexed pages whose OCR failed, emitted as a flow
            list (``[2, 5]``) only when not None.
    """
    metadata: dict[str, object] = {
        "file_hash": file_hash,
        "raw_file_path": _QuotedStr(raw_file_path),
        "conversion_date": _QuotedStr(datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")),
        "conversion_method": conversion_method,
        "extractor": _QuotedStr(extractor),
    }
    if ocr_failed_pages is not None:
        metadata["ocr_failed_pages"] = ocr_failed_pages
    return (
        "---\n"
        + yaml.safe_dump(
            metadata,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
            default_flow_style=None,  # inline lists of scalars: [2, 5]
        )
        + "---\n\n"
    )


def image_to_data_uri(image: Image.Image, max_dimension: int | None = None) -> str:
    """Encode a PIL image as a base64 PNG data URI.

    If max_dimension is set, the image is downscaled (never upscaled) so its
    longest edge fits within it, preserving aspect ratio.
    """
    if max_dimension is not None:
        image = image.copy()
        image.thumbnail((max_dimension, max_dimension))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# Args whose values determine output content and are included in the version hash.
# Changing any other arg (IO paths, --force, --verbose, --image-cache, --concurrency,
# --max-retries) must NOT alter the hash so that output directories remain stable
# across unrelated changes.
# GENERATION_ARGS: frozenset[str] = frozenset({
#     "model", "max_tokens", "temperature", "top_p", "target_px", "conversion_method"
# })


# def compute_generation_hash(args: argparse.Namespace, length: int = 8) -> str:
#     """Return a short stable hash over generation-specific args for output versioning."""
#     gen_dict = {k: v for k, v in vars(args).items()}
#     # gen_dict["_renderer"] = "pdf2image"
#     return hashlib.md5(json.dumps(gen_dict, sort_keys=True).encode()).hexdigest()[:length]
