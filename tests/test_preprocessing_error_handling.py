"""Offline tests for the deliberately open ``except Exception`` batch guards.

Each guard in ``doc2md`` must (1) swallow the error and
return its documented fallback, and (2) tally the exception type via
``record_failure`` so real runs reveal which types occur. Failures are
injected with fakes/monkeypatching — no network, GPU, or model calls.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from PIL import Image

from doc2md import caption as caption_mod
from doc2md import convert as convert_mod
from doc2md import resolve as resolve_mod
from doc2md import utils

PLACEHOLDER = "![image](image_1.png)100,100,500,500"
INVERTED_BBOX = "![image](image_1.png)900,900,100,100"


@pytest.fixture(autouse=True)
def clear_failure_counts():
    utils.FAILURE_COUNTS.clear()
    yield
    utils.FAILURE_COUNTS.clear()


def write_page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 300), (10, 20, 30)).save(path)


def test_record_failure_tallies_by_stage_and_type():
    detail = utils.record_failure("crop", ValueError("bad"))
    utils.record_failure("crop", ValueError("worse"))
    utils.record_failure("ocr", TimeoutError())

    assert detail == "ValueError: bad"
    assert utils.FAILURE_COUNTS[("crop", "ValueError")] == 2
    assert utils.FAILURE_COUNTS[("ocr", "TimeoutError")] == 1


def test_log_failure_summary_is_silent_when_empty(capsys):
    utils.log_failure_summary()
    assert capsys.readouterr().out == ""


# --- resolve_bbox_images: crop guard ---------------------------------------


@pytest.mark.parametrize(
    ("placeholder", "images_dir_exists", "expected_type"),
    [
        (INVERTED_BBOX, True, "ValueError"),  # PIL rejects right < left
        (PLACEHOLDER, False, "FileNotFoundError"),  # save into missing dir
    ],
)
def test_resolve_section_crop_failure_keeps_placeholder(
    tmp_path, placeholder, images_dir_exists, expected_type
):
    cache = tmp_path / "cache"
    write_page(cache / "doc" / "p000001-000001.png")
    images_dir = tmp_path / "images"
    if images_dir_exists:
        images_dir.mkdir()

    text = f"<!-- page 1 -->\n{placeholder}"
    updated, n_resolved, n_failed = resolve_mod.resolve_section(
        text, 1, "doc", cache, images_dir, embed_base64=False
    )

    assert (updated, n_resolved, n_failed) == (text, 0, 1)
    assert utils.FAILURE_COUNTS[("crop", expected_type)] == 1


def test_resolve_section_corrupt_page_image_is_not_swallowed(tmp_path):
    """Image.open sits outside the guard, so a corrupt page PNG propagates
    to the per-file guard rather than being tallied as a crop failure."""
    cache = tmp_path / "cache"
    corrupt = cache / "doc" / "p000001-000001.png"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not a png")

    with pytest.raises(Image.UnidentifiedImageError):
        resolve_mod.resolve_section(
            f"<!-- page 1 -->\n{PLACEHOLDER}",
            1,
            "doc",
            cache,
            tmp_path,
            embed_base64=False,
        )


def test_resolve_main_file_failure_is_tallied_and_run_continues(tmp_path, monkeypatch):
    input_dir = tmp_path / "md"
    (input_dir / "page_image_cache").mkdir(parents=True)
    (input_dir / "bad.md").write_bytes(b"\xff\xfe\x00 not utf-8 \x80")
    (input_dir / "good.md").write_text("no placeholders", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["resolve", "-i", str(input_dir)])

    resolve_mod.main()

    assert utils.FAILURE_COUNTS[("resolve_file", "UnicodeDecodeError")] == 1
    assert (input_dir / "resolved" / "good.md").exists()
    assert not (input_dir / "resolved" / "bad.md").exists()


# --- convert: ocr / anydoc / convert_file guards ------------


class RaisingOCRClient:
    """Duck-typed httpx.AsyncClient whose post() raises a chosen exception."""

    def __init__(self, exc: Exception):
        self.exc = exc

    async def post(self, url, json):
        raise self.exc


@pytest.mark.parametrize(
    "exc",
    [KeyError("choices"), RuntimeError("boom"), OSError("socket closed")],
)
def test_convert_pages_vlm_tallies_unexpected_error_types(tmp_path, exc):
    page = tmp_path / "p000001-000001.png"
    write_page(page)

    result = asyncio.run(
        convert_mod.convert_pages_vlm(
            [page],
            RaisingOCRClient(exc),
            asyncio.Semaphore(1),
            base_url="http://fake",
            model="m",
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            target_px=64,
            max_retries=1,
            source_name="doc",
        )
    )

    assert "<!-- OCR FAILED:" in result.markdown
    assert result.failed_pages == [1]
    assert utils.FAILURE_COUNTS[("ocr", type(exc).__qualname__)] == 1


@pytest.mark.parametrize(
    ("exc", "expected_stage"),
    [
        (convert_mod.anydoc.MalformedError("bad zip"), "anydoc"),
        (convert_mod.anydoc.EncryptedError("locked"), "anydoc_encrypted"),
        (convert_mod.anydoc.UnsupportedError("unknown"), "anydoc_unsupported"),
        (convert_mod.anydoc.NeedsOcrError("all 2 pages need OCR"), "anydoc_needs_ocr"),
    ],
)
def test_process_one_file_native_tallies_anydoc_error_by_kind(
    tmp_path, monkeypatch, exc, expected_stage
):
    def explode(path: str) -> str:
        raise exc

    monkeypatch.setattr(convert_mod.anydoc, "to_markdown", explode)

    result = asyncio.run(
        convert_mod.process_one_file(
            tmp_path / "doc.docx",
            "abc123",
            [],
            tmp_path,
            client=None,
            sem=asyncio.Semaphore(1),
            args=None,
            method="native",
        )
    )

    assert result == {"status": "failed", "name": "doc.docx"}
    assert not (tmp_path / "abc123.md").exists()
    assert utils.FAILURE_COUNTS[(expected_stage, type(exc).__qualname__)] == 1


def test_process_one_file_tallies_and_reports_failure(tmp_path, monkeypatch):
    async def explode(*args, **kwargs):
        raise PermissionError("read-only input")

    monkeypatch.setattr(convert_mod, "convert_native", explode)

    result = asyncio.run(
        convert_mod.process_one_file(
            Path("doc.pdf"),
            "abc123",
            [],
            tmp_path,
            client=None,
            sem=asyncio.Semaphore(1),
            args=None,
            method="native",
        )
    )

    assert result == {"status": "failed", "name": "doc.pdf"}
    assert utils.FAILURE_COUNTS[("convert_file", "PermissionError")] == 1


# --- generate_image_captions: caption guard --------------------------------


class RaisingCaptionClient:
    """Duck-typed instructor client whose create() raises a chosen error."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        raise self.exc


@pytest.mark.parametrize("exc", [TimeoutError("slow"), ValueError("schema mismatch")])
def test_caption_one_image_returns_none_and_tallies(tmp_path, exc):
    image_path = tmp_path / "img.png"
    write_page(image_path)

    result = asyncio.run(
        caption_mod.caption_one_image(
            RaisingCaptionClient(exc),
            image_path,
            model="hosted_vllm/m",
            base_url="http://fake",
            api_key=None,
            max_tokens=8,
            temperature=0.0,
            max_image_dimension=64,
            max_retries=1,
            sem=asyncio.Semaphore(1),
        )
    )

    assert result is None
    assert utils.FAILURE_COUNTS[("caption", type(exc).__qualname__)] == 1


def test_caption_one_image_missing_file_is_tallied(tmp_path):
    result = asyncio.run(
        caption_mod.caption_one_image(
            RaisingCaptionClient(RuntimeError("unused")),
            tmp_path / "missing.png",
            model="hosted_vllm/m",
            base_url="http://fake",
            api_key=None,
            max_tokens=8,
            temperature=0.0,
            max_image_dimension=64,
            max_retries=1,
            sem=asyncio.Semaphore(1),
        )
    )

    assert result is None
    assert utils.FAILURE_COUNTS[("caption", "FileNotFoundError")] == 1
