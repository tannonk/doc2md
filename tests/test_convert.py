"""Offline tests for the async PDF/TIFF/MS-Office-to-Markdown conversion stage."""

import argparse
import asyncio
import base64
import importlib.metadata
import io
import shutil
import subprocess
import sys
from pathlib import Path

import anydoc
import httpx
import pytest
from PIL import Image

from doc2md import convert as convert_mod
from doc2md import utils
from doc2md.convert import (
    ANYDOC_EXTRACTOR,
    NativeUnsupportedError,
    _needs_ocr_check,
    _summarize_pages,
    aggregate_stats,
    build_front_matter,
    compute_file_hash,
    convert_native,
    convert_pages_vlm,
    explain_native_failure,
    ocr_page,
    process_one_file,
    render_pages,
    resolve_conversion_method,
)

requires_poppler = pytest.mark.skipif(
    shutil.which("pdftoppm") is None, reason="poppler (pdftoppm) not on PATH"
)
requires_libreoffice = pytest.mark.skipif(
    shutil.which("soffice") is None, reason="LibreOffice (soffice) not on PATH"
)


def _generation_namespace(**overrides) -> argparse.Namespace:
    """legacy compute_generation_hash reads vars(args), i.e. an instance __dict__ —
    a plain class with class-level-only attributes returns {} from vars(),
    so a real Namespace is needed here, matching what argparse actually
    produces in main()."""
    defaults = {
        "model": "fake-model",
        "max_tokens": 64,
        "temperature": 0.2,
        "top_p": 0.9,
        "target_px": 1540,
        "conversion_method": "auto",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}", request=None, response=self
            )

    def json(self) -> dict:
        return self._body


class FakeOCRClient:
    """Duck-typed stand-in for httpx.AsyncClient: implements only post().

    fail_times: number of leading calls that raise a transient error before
    succeeding. always_fail_status: if set, every call raises with this
    status code instead of ever succeeding.
    """

    def __init__(
        self,
        fail_times: int = 0,
        always_fail_status: int | None = None,
        text: str = "OCR text",
    ):
        self.fail_times = fail_times
        self.always_fail_status = always_fail_status
        self.text = text
        self.call_count = 0

    async def post(self, url, json):
        self.call_count += 1
        if self.always_fail_status is not None:
            return FakeResponse(self.always_fail_status, {})
        if self.call_count <= self.fail_times:
            raise httpx.TimeoutException("simulated timeout")
        return FakeResponse(200, {"choices": [{"message": {"content": self.text}}]})


def _semaphore():
    return asyncio.Semaphore(4)


def test_ocr_page_returns_text_on_success():
    client = FakeOCRClient(text="hello world")
    result = asyncio.run(
        ocr_page(client, "b64data", base_url="http://fake", model="m", max_retries=2)
    )
    assert result == "hello world"
    assert client.call_count == 1


def test_ocr_page_retries_transient_errors_then_succeeds():
    client = FakeOCRClient(fail_times=2, text="recovered")
    result = asyncio.run(
        ocr_page(client, "b64data", base_url="http://fake", model="m", max_retries=3)
    )
    assert result == "recovered"
    assert client.call_count == 3


def test_ocr_page_does_not_retry_4xx():
    client = FakeOCRClient(always_fail_status=400)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(
            ocr_page(
                client, "b64data", base_url="http://fake", model="m", max_retries=3
            )
        )
    assert client.call_count == 1


def test_ocr_page_raises_after_retries_exhausted():
    client = FakeOCRClient(always_fail_status=503)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(
            ocr_page(
                client, "b64data", base_url="http://fake", model="m", max_retries=3
            )
        )
    assert client.call_count == 3


def _write_page(path, size=(300, 400), color=(10, 20, 30)):
    Image.new("RGB", size, color=color).save(path, "PNG")


def test_convert_pages_vlm_downscales_before_sending(tmp_path):
    page_path = tmp_path / "p001-001.png"
    _write_page(page_path, size=(850, 1100))

    captured = {}

    async def _capture_post(url, json):
        data_uri = json["messages"][0]["content"][0]["image_url"]["url"]
        b64_payload = data_uri.split(",", 1)[1]
        sent_img = Image.open(io.BytesIO(base64.b64decode(b64_payload)))
        captured["sent_size"] = sent_img.size
        return FakeResponse(200, {"choices": [{"message": {"content": "page text"}}]})

    client = FakeOCRClient()
    client.post = _capture_post

    markdown = asyncio.run(
        convert_pages_vlm(
            [page_path],
            client,
            _semaphore(),
            base_url="http://fake",
            model="m",
            max_tokens=64,
            temperature=0.2,
            top_p=0.9,
            target_px=256,
            max_retries=2,
            source_name="doc",
        )
    ).markdown

    assert "<!-- page 1 -->" in markdown
    assert "page text" in markdown
    assert max(captured["sent_size"]) <= 256
    assert captured["sent_size"] != (850, 1100)


def test_convert_pages_vlm_returns_placeholder_on_failure(tmp_path):
    page_path = tmp_path / "p001-001.png"
    _write_page(page_path, size=(100, 100))
    client = FakeOCRClient(always_fail_status=500)

    result = asyncio.run(
        convert_pages_vlm(
            [page_path],
            client,
            _semaphore(),
            base_url="http://fake",
            model="m",
            max_tokens=64,
            temperature=0.2,
            top_p=0.9,
            target_px=64,
            max_retries=1,
            source_name="doc",
        )
    )

    assert "<!-- page 1 -->" in result.markdown
    assert "<!-- OCR FAILED:" in result.markdown
    assert result.failed_pages == [1]


def test_convert_pages_vlm_preserves_page_order_under_concurrency(tmp_path):
    page_paths = []
    for i in range(3):
        p = tmp_path / f"p{i + 1:03d}-003.png"
        _write_page(p, size=(100, 100), color=(i * 30, 0, 0))
        page_paths.append(p)

    class DelayedClient(FakeOCRClient):
        async def post(self, url, json):
            # later pages "finish" first to prove gather still preserves order
            self.call_count += 1
            await asyncio.sleep(0.03 * (4 - self.call_count))
            return FakeResponse(
                200, {"choices": [{"message": {"content": f"text-{self.call_count}"}}]}
            )

    markdown = asyncio.run(
        convert_pages_vlm(
            page_paths,
            DelayedClient(),
            asyncio.Semaphore(4),
            base_url="http://fake",
            model="m",
            max_tokens=64,
            temperature=0.2,
            top_p=0.9,
            target_px=128,
            max_retries=1,
            source_name="doc",
        )
    ).markdown

    idx1 = markdown.index("<!-- page 1 -->")
    idx2 = markdown.index("<!-- page 2 -->")
    idx3 = markdown.index("<!-- page 3 -->")
    assert idx1 < idx2 < idx3


def _make_docx(path, text="Hello from a synthesized docx fixture."):
    import docx

    document = docx.Document()
    document.add_paragraph(text)
    document.save(path)


def test_convert_native_extracts_text_via_anydoc(tmp_path):
    docx_path = tmp_path / "doc.docx"
    _make_docx(docx_path, text="Unique marker sentence for anydoc extraction test.")

    markdown = asyncio.run(convert_native(docx_path))

    assert "Unique marker sentence for anydoc extraction test." in markdown


def test_convert_native_raises_typed_anydoc_error(tmp_path):
    bogus_path = tmp_path / "doc.docx"
    bogus_path.write_bytes(b"not actually a docx file")

    with pytest.raises(anydoc.MalformedError):
        asyncio.run(convert_native(bogus_path))


def test_convert_native_rejects_tiff_without_calling_anydoc(tmp_path, monkeypatch):
    def must_not_be_called(path: str) -> str:
        raise AssertionError("anydoc must not be called for TIFF input")

    monkeypatch.setattr(anydoc, "to_markdown", must_not_be_called)
    tif_path = tmp_path / "scan.tiff"
    _make_multiframe_tiff(tif_path, n_pages=1)

    with pytest.raises(NativeUnsupportedError):
        asyncio.run(convert_native(tif_path))


def _make_multipage_pdf(path, n_pages: int, size=(200, 300)):
    images = [
        Image.new("RGB", size, color=(i * 30 % 256, 0, 0)) for i in range(n_pages)
    ]
    images[0].save(path, format="PDF", save_all=True, append_images=images[1:])


def _make_multiframe_tiff(path, n_pages: int, size=(200, 300)):
    frames = [
        Image.new("RGB", size, color=(0, i * 30 % 256, 0)) for i in range(n_pages)
    ]
    frames[0].save(path, format="TIFF", save_all=True, append_images=frames[1:])


@requires_poppler
def test_render_pages_pdf_caches_full_resolution(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    _make_multipage_pdf(pdf_path, n_pages=3)
    cache_dir = tmp_path / "cache"

    paths = render_pages(pdf_path, cache_dir, "abc123", dpi=100)

    assert len(paths) == 3
    assert [p.name for p in paths] == [
        "p000001-000003.png",
        "p000002-000003.png",
        "p000003-000003.png",
    ]
    for p in paths:
        assert p.parent == cache_dir / "abc123"
        assert p.exists()


def test_render_pages_tiff_caches_native_resolution(tmp_path):
    tif_path = tmp_path / "doc.tif"
    _make_multiframe_tiff(tif_path, n_pages=2, size=(400, 500))
    cache_dir = tmp_path / "cache"

    paths = render_pages(tif_path, cache_dir, "abc123")

    assert len(paths) == 2
    for p in paths:
        cached = Image.open(p)
        assert cached.size == (400, 500)  # native resolution, not pre-downscaled


@requires_libreoffice
def test_render_pages_office_converts_and_caches(tmp_path):
    docx_path = tmp_path / "doc.docx"
    _make_docx(docx_path)
    cache_dir = tmp_path / "cache"

    paths = render_pages(docx_path, cache_dir, "abc123", dpi=100)

    assert len(paths) >= 1
    assert paths[0].parent == cache_dir / "abc123"
    assert paths[0].exists()


@requires_libreoffice
def test_render_pages_office_failure_raises(tmp_path):
    # LibreOffice's Writer import is too permissive to reject garbage bytes
    # (it renders them as raw text), but it does still fail — silently, with
    # exit code 0 — when the source file doesn't exist at all. This exercises
    # convert_office_to_pdf's output-file-exists check, which is exactly what
    # catches that silent-exit-0 failure mode.
    missing_path = tmp_path / "does-not-exist.docx"
    cache_dir = tmp_path / "cache"

    with pytest.raises(RuntimeError):
        render_pages(missing_path, cache_dir, "abc123")


def test_render_pages_unsupported_suffix_raises(tmp_path):
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("plain text")

    with pytest.raises(ValueError):
        render_pages(txt_path, tmp_path / "cache", "abc123")


def test_resolve_conversion_method_auto_picks_per_format_default():
    assert resolve_conversion_method(Path("doc.docx"), "auto") == "native"
    assert resolve_conversion_method(Path("doc.pptx"), "auto") == "native"
    assert resolve_conversion_method(Path("doc.pdf"), "auto") == "vlm"
    assert resolve_conversion_method(Path("doc.tif"), "auto") == "vlm"


def test_resolve_conversion_method_explicit_override_wins():
    assert resolve_conversion_method(Path("doc.docx"), "vlm") == "vlm"
    assert resolve_conversion_method(Path("doc.pdf"), "native") == "native"


def test_needs_ocr_check():
    pdf_files = [Path("a.pdf"), Path("b.pdf")]
    office_files = [Path("a.docx"), Path("b.pptx")]
    mixed_files = [Path("a.pdf"), Path("b.docx")]

    assert _needs_ocr_check(pdf_files, "auto") is True
    assert _needs_ocr_check(office_files, "auto") is False
    assert _needs_ocr_check(office_files, "native") is False
    assert _needs_ocr_check(office_files, "vlm") is True
    assert _needs_ocr_check(mixed_files, "auto") is True


def test_process_one_file_vlm_dispatch(tmp_path):
    page_path = tmp_path / "p001-001.png"
    _write_page(page_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    args = _generation_namespace(
        conversion_method="vlm", base_url="http://fake", max_retries=1, concurrency=4
    )
    input_file = Path("doc.pdf")

    result = asyncio.run(
        process_one_file(
            input_file,
            "abc123",
            [page_path],
            output_dir,
            FakeOCRClient(text="hi"),
            _semaphore(),
            args,
            method="vlm",
        )
    )

    assert result == {"status": "converted", "name": "doc.pdf"}
    written = (output_dir / "abc123.md").read_text(encoding="utf-8")
    assert "<!-- page 1 -->" in written
    assert "hi" in written
    assert written.startswith("---\n")
    assert "file_hash: abc123" in written
    assert 'raw_file_path: "doc.pdf"' in written
    assert "conversion_method: vlm" in written
    assert 'extractor: "fake-model"' in written
    assert "ocr_failed_pages" not in written
    assert "ocr_model" not in written


def test_process_one_file_native_dispatch(tmp_path):
    docx_path = tmp_path / "doc.docx"
    _make_docx(docx_path, text="Native dispatch marker text.")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    args = _generation_namespace(conversion_method="auto")

    result = asyncio.run(
        process_one_file(
            docx_path,
            "def456",
            [],
            output_dir,
            FakeOCRClient(),
            _semaphore(),
            args,
            method="native",
        )
    )

    assert result == {"status": "converted", "name": "doc.docx"}
    written = (output_dir / "def456.md").read_text(encoding="utf-8")
    assert "Native dispatch marker text." in written
    assert "file_hash: def456" in written
    assert 'raw_file_path: "' in written and "doc.docx" in written
    assert "conversion_method: native" in written
    assert f'extractor: "{ANYDOC_EXTRACTOR}"' in written
    assert "ocr_model" not in written


def test_compute_file_hash_is_deterministic_and_content_based(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"identical content")
    b.write_bytes(b"identical content")
    c = tmp_path / "c.txt"
    c.write_bytes(b"different content")

    assert compute_file_hash(a) == compute_file_hash(b)  # same content, different names
    assert compute_file_hash(a) == compute_file_hash(a)  # deterministic across calls
    assert compute_file_hash(a) != compute_file_hash(c)


def test_compute_file_hash_respects_length(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"some content")

    assert len(compute_file_hash(path)) == 12  # default
    assert len(compute_file_hash(path, length=6)) == 6


def test_build_front_matter():
    block = build_front_matter(
        file_hash="abc123",
        raw_file_path="Weird Name (final) — v2.docx",
        conversion_method="native",
        extractor="anydoc v0.2.4",
    )

    assert block.startswith("---\n")
    assert block.rstrip("\n").endswith("---")
    lines = [line for line in block.splitlines() if ": " in line]
    keys = [line.split(":", 1)[0] for line in lines]

    assert keys == [
        "file_hash",
        "raw_file_path",
        "conversion_date",
        "conversion_method",
        "extractor",
    ]
    assert 'extractor: "anydoc v0.2.4"' in block
    assert "abc123" in block
    assert "Weird Name (final) — v2.docx" in block
    assert "native" in block


def test_aggregate_stats_counts_by_status():
    results = [
        {"status": "converted", "name": "a"},
        {"status": "failed", "name": "b"},
        {"status": "converted", "name": "c"},
        {"status": "partial", "name": "d"},
    ]
    assert aggregate_stats(results) == {"converted": 2, "partial": 1, "failed": 1}


def test_aggregate_stats_empty():
    assert aggregate_stats([]) == {"converted": 0, "partial": 0, "failed": 0}


# --- conversion-method routing and failure semantics ------------------------


@pytest.fixture
def clear_failure_counts():
    utils.FAILURE_COUNTS.clear()
    yield
    utils.FAILURE_COUNTS.clear()


@pytest.mark.parametrize(
    "name", ["a.odt", "a.rtf", "a.xls", "a.xlsx", "a.ods", "a.pptm", "a.odp"]
)
def test_resolve_conversion_method_routes_office_variants_to_native(name):
    assert resolve_conversion_method(Path(name), "auto") == "native"
    assert resolve_conversion_method(Path(name), "vlm") == "vlm"


def test_summarize_pages_compresses_runs():
    assert _summarize_pages([1, 2, 3, 4]) == "1-4"
    assert _summarize_pages([1, 2, 5, 7, 8]) == "1-2, 5, 7-8"
    assert _summarize_pages([3]) == "3"


def test_explain_native_failure_for_scanned_pdf(tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    _make_multipage_pdf(pdf_path, n_pages=2)  # image-only pages, no text layer

    with pytest.raises(anydoc.NeedsOcrError) as excinfo:
        anydoc.to_markdown(str(pdf_path))
    stage, reason = explain_native_failure(excinfo.value)

    assert stage == "anydoc_needs_ocr"
    assert "page(s) 1-2 of 2" in reason
    assert "--conversion-method auto or vlm" in reason


def test_process_one_file_native_scanned_pdf_fails_without_output(
    tmp_path, clear_failure_counts
):
    pdf_path = tmp_path / "scan.pdf"
    _make_multipage_pdf(pdf_path, n_pages=1)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    result = asyncio.run(
        process_one_file(
            pdf_path,
            "scan01",
            [],
            output_dir,
            FakeOCRClient(),
            _semaphore(),
            _generation_namespace(conversion_method="native"),
            method="native",
        )
    )

    assert result == {"status": "failed", "name": "scan.pdf"}
    assert not (output_dir / "scan01.md").exists()
    assert utils.FAILURE_COUNTS[("anydoc_needs_ocr", "NeedsOcrError")] == 1


class PageSelectiveClient(FakeOCRClient):
    """Fails (non-retryable 400) for page images whose top-left pixel is red."""

    async def post(self, url, json):
        self.call_count += 1
        data_uri = json["messages"][0]["content"][0]["image_url"]["url"]
        image = Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1])))
        if image.convert("RGB").getpixel((0, 0)) == (255, 0, 0):
            return FakeResponse(400, {})
        return FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})


def _vlm_pages(tmp_path, colors):
    paths = []
    for i, color in enumerate(colors):
        path = tmp_path / f"p{i + 1:03d}-{len(colors):03d}.png"
        _write_page(path, size=(64, 64), color=color)
        paths.append(path)
    return paths


def _run_vlm_file(tmp_path, page_paths):
    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True)
    args = _generation_namespace(base_url="http://fake", max_retries=1)
    result = asyncio.run(
        process_one_file(
            Path("doc.pdf"),
            "vlm001",
            page_paths,
            output_dir,
            PageSelectiveClient(),
            _semaphore(),
            args,
            method="vlm",
        )
    )
    return result, output_dir / "vlm001.md"


def test_process_one_file_vlm_partial_failure_is_flagged(tmp_path):
    red, grey = (255, 0, 0), (90, 90, 90)
    pages = _vlm_pages(tmp_path, [grey, red, grey])

    result, output_path = _run_vlm_file(tmp_path, pages)

    assert result == {"status": "partial", "name": "doc.pdf"}
    written = output_path.read_text(encoding="utf-8")
    assert "ocr_failed_pages: [2]" in written
    assert "<!-- OCR FAILED:" in written


def test_process_one_file_vlm_all_pages_failed_writes_nothing(tmp_path):
    red = (255, 0, 0)
    result, output_path = _run_vlm_file(tmp_path, _vlm_pages(tmp_path, [red, red]))

    assert result == {"status": "failed", "name": "doc.pdf"}
    assert not output_path.exists()


def test_process_one_file_vlm_without_pages_writes_nothing(tmp_path):
    result, output_path = _run_vlm_file(tmp_path, [])

    assert result == {"status": "failed", "name": "doc.pdf"}
    assert not output_path.exists()


# --- main(): end-to-end routing --------------------------------------------


def _run_main(monkeypatch, input_dir, output_dir, *extra_args):
    monkeypatch.setattr(
        sys,
        "argv",
        ["convert", "-i", str(input_dir), "-o", str(output_dir), *extra_args],
    )
    convert_mod.main()


def _outputs(output_dir):
    return sorted(p.name for p in output_dir.glob("*.md"))


def test_main_native_never_renders_and_explains_failures(
    tmp_path, monkeypatch, capsys, clear_failure_counts
):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    _make_docx(input_dir / "report.docx", text="Native main marker.")
    _make_multipage_pdf(input_dir / "scan.pdf", n_pages=2)
    _make_multiframe_tiff(input_dir / "fax.tif", n_pages=1)

    def must_not_render(*args, **kwargs):
        raise AssertionError("render_pages must not run for native files")

    def must_not_check(*args, **kwargs):
        raise AssertionError("no vLLM check for a native-only batch")

    monkeypatch.setattr(convert_mod, "render_pages", must_not_render)
    monkeypatch.setattr(convert_mod, "check_model_available", must_not_check)

    _run_main(monkeypatch, input_dir, output_dir, "--conversion-method", "native")

    docx_hash = compute_file_hash(input_dir / "report.docx")
    assert _outputs(output_dir) == [f"{docx_hash}.md"]
    log = capsys.readouterr().out
    assert "can't be converted natively" in log and "fax.tif" in log
    assert "need OCR" in log
    assert "1 converted, 0 partial, 0 skipped, 2 failed" in log
    assert utils.FAILURE_COUNTS[("anydoc_needs_ocr", "NeedsOcrError")] == 1
    assert utils.FAILURE_COUNTS[("anydoc_unsupported", "NativeUnsupportedError")] == 1

    # failed files wrote nothing, so a rerun retries them and skips the success
    _run_main(monkeypatch, input_dir, output_dir, "--conversion-method", "native")
    assert "0 converted, 0 partial, 1 skipped, 2 failed" in capsys.readouterr().out


# def test_main_defaults_to_auto_routing(tmp_path, monkeypatch, capsys):
#     input_dir, output_dir = tmp_path / "in", tmp_path / "out"
#     input_dir.mkdir()
#     _make_docx(input_dir / "report.docx")
#     _write_page(input_dir / "scan.tif", size=(64, 64))

#     rendered: list[str] = []

#     def fake_render(input_file, cache_dir, file_hash, dpi=200):
#         rendered.append(input_file.name)
#         page = cache_dir / file_hash / "p001-001.png"
#         page.parent.mkdir(parents=True, exist_ok=True)
#         _write_page(page, size=(64, 64))
#         return [page]

#     async def fake_ocr_page(client, image_b64, **kwargs):
#         return "vlm text"

#     monkeypatch.setattr(convert_mod, "render_pages", fake_render)
#     monkeypatch.setattr(convert_mod, "check_model_available", lambda *a: None)
#     monkeypatch.setattr(convert_mod, "ocr_page", fake_ocr_page)

#     _run_main(monkeypatch, input_dir, output_dir)

#     assert rendered == ["scan.tif"]  # the docx went to anydoc, not LibreOffice
#     tif_md = output_dir / f"{compute_file_hash(input_dir / 'scan.tif')}.md"
#     docx_md = output_dir / f"{compute_file_hash(input_dir / 'report.docx')}.md"
#     assert "conversion_method: vlm" in tif_md.read_text(encoding="utf-8")
#     assert "conversion_method: native" in docx_md.read_text(encoding="utf-8")
#     assert '"conversion_method": "auto"' in (output_dir / "config.json").read_text()
#     assert "2 converted" in capsys.readouterr().out


@pytest.mark.parametrize(
    "render_error",
    [
        subprocess.TimeoutExpired(cmd="soffice", timeout=120),
        FileNotFoundError(2, "No such file or directory", "soffice"),
        OSError("cannot write mode CMYK as PNG"),
    ],
)
def test_main_vlm_render_failure_is_isolated(
    tmp_path, monkeypatch, capsys, clear_failure_counts, render_error
):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    _make_docx(input_dir / "broken.docx")
    _write_page(input_dir / "ok.tif", size=(64, 64))

    def flaky_render(input_file, cache_dir, file_hash, dpi=200):
        if input_file.suffix == ".docx":
            raise render_error
        page = cache_dir / file_hash / "p001-001.png"
        page.parent.mkdir(parents=True, exist_ok=True)
        _write_page(page, size=(64, 64))
        return [page]

    async def fake_ocr_page(client, image_b64, **kwargs):
        return "vlm text"

    monkeypatch.setattr(convert_mod, "render_pages", flaky_render)
    monkeypatch.setattr(convert_mod, "check_model_available", lambda *a: None)
    monkeypatch.setattr(convert_mod, "ocr_page", fake_ocr_page)

    _run_main(monkeypatch, input_dir, output_dir, "--conversion-method", "vlm")

    log = capsys.readouterr().out
    assert "1 converted, 0 partial, 0 skipped, 1 failed" in log
    assert "broken.docx: rendering failed" in log
    if isinstance(render_error, FileNotFoundError):
        assert "setup_libreoffice.sh" in log
    assert utils.FAILURE_COUNTS[("render", type(render_error).__qualname__)] == 1


def test_anydoc_extractor_label_includes_installed_version():
    assert (
        ANYDOC_EXTRACTOR == f"anydoc v{importlib.metadata.version('firecrawl-anydoc')}"
    )


def test_build_front_matter_lists_failed_pages_after_extractor():
    block = build_front_matter(
        file_hash="abc123",
        raw_file_path="doc.pdf",
        conversion_method="vlm",
        extractor="lightonai/LightOnOCR-2-1B",
        ocr_failed_pages=[2, 5],
    )

    assert 'extractor: "lightonai/LightOnOCR-2-1B"\nocr_failed_pages: [2, 5]\n' in block
