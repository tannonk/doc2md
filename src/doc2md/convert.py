"""Convert PDFs, TIFFs, and MS Office / OpenDocument files to Markdown.

Async processing handles conversion across many files.

Every input file is routed to one of two backends by --conversion-method
(see resolve_conversion_method):

    method    PDF                    TIFF                  Office / OpenDocument
    --------  ---------------------  --------------------  -----------------------
    auto      VLM                    VLM                   anydoc
    vlm (*)   VLM                    VLM                   LibreOffice -> PDF -> VLM
    native    anydoc (text layer     fails: anydoc has     anydoc
              only, see below)       no TIFF support
    (*) default

Office / OpenDocument covers word-processing (.doc/.docx/.docm/.dotx/.dotm/
.odt/.rtf), spreadsheet (.xls/.xlsx/.xlsm/.xltx/.ods) and presentation
(.ppt/.pptx/.pptm/.ppsx/.potx/.odp) files — see OFFICE_SUFFIXES.

Processing is split into two stages that run one after the other:

  1. render_pages()
    - Renders and caches a full-resolution PNG per page/frame/slide
        (page_image_cache/<file_hash>/pNNN-MMM.png). PDF pages go through
        pdf2image (poppler's pdftoppm run as a subprocess); TIFF frames are read
        directly; Office documents are converted to PDF via headless LibreOffice
        first, then rendered the same way as PDF. 
        This stage is synchronous and runs in the main thread.
        Files routed to "native" skip this stage entirely.

  2. Markdown conversion — async, per file:
    - "vlm": OCRs the cached pages via a vLLM server, concurrently across
        files and pages, gated by a single semaphore shared across the whole
        run (--concurrency). Format-agnostic. A PDF page, a TIFF frame, and
        a LibreOffice-rendered Office page are all treated as a PNG on disk.
    - "native": anydoc.to_markdown() run directly against the original
        document. Warning this does NOT use OCR, and doesn't generate any 
        page images! As a result, scanned (image-only) PDF raises 
        anydoc.NeedsOcrError, which is reported as a failure naming 
        the pages that need OCR and pointing at auto/vlm.

Failures are isolated and never abort the batch:
  - A page that fails OCR becomes an <!-- OCR FAILED: ... --> placeholder.
    The document is still written, lists the pages in its front matter
    (ocr_failed_pages), and counts as "partial"; re-run with --force to retry.
  - A document that can't be converted at all (render failure, every VLM page
    failed, or an anydoc error) writes NO .md and counts as "failed", so the
    next run retries it automatically.

To handle arbitrary file names, we compute a content hash for each input file 
and use that hash to name both the output Markdown file (<file_hash>.md) and its
page-image cache subdirectory (page_image_cache/<file_hash>/).
The original filename is preserved for traceability in a YAML front-matter block prepended
to the output file: file_hash/raw_file_path/conversion_date/conversion_method/
extractor, where extractor is the VLM model id for "vlm" output or
"anydoc v<version>" for "native" output, plus ocr_failed_pages when some VLM
pages failed.

External requirements depend on which files are routed to "vlm":
  - a vLLM server serving --model (checked up front; skipped when nothing in
    the batch is routed to "vlm").
  - poppler-utils (pdftoppm) on PATH for PDFs and Office files routed to "vlm":
        conda install -c conda-forge poppler
    (or `apt-get install poppler-utils` / `brew install poppler`)
  - LibreOffice (soffice) on PATH only for Office files routed to "vlm" (i.e.
    --conversion-method vlm) — see scripts/setup_libreoffice.sh.

PDF and Office pages are cached at a fixed 200 DPI (DEFAULT_CACHE_DPI).
The --target-px flag controls the resolution of the downscaled copy sent 
to the OCR model on the "vlm" path. TIFF frames are cached at their
native resolution.

Example usage:

    python -m doc2md.convert \
        -i <path-to-dir-containing-raw-documents> \
        -o <path-to-dir-for-markdown-output> \
        --verbose
"""

import argparse
import asyncio
import base64
import io
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from urllib.parse import urljoin

import anydoc
import httpx
from loguru import logger
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PopplerNotInstalledError
from PIL import Image, ImageSequence
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from doc2md.utils import (
    build_front_matter,
    check_model_available,
    compute_file_hash,
    log_failure_summary,
    record_failure,
    serialize_args,
)

DEFAULT_BASE_URL = "http://localhost:8002"
DEFAULT_MODEL = "lightonai/LightOnOCR-2-1B-bbox-soup"
DEFAULT_CONVERSION_METHOD = "vlm"
CONVERSION_METHODS: tuple[str, ...] = ("auto", "vlm", "native")

DEFAULT_REQUEST_TIMEOUT_S = 300  # generous: a full-page OCR decode can be slow
DEFAULT_LIBREOFFICE_TIMEOUT_S = 120  # per-document LibreOffice conversion timeout
DEFAULT_CACHE_DPI = 200  # fixed rendering resolution for cached PDF/Office

TIFF_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff"})
# Born-digital formats that both anydoc ("native") and LibreOffice ("vlm") read.
OFFICE_SUFFIXES: frozenset[str] = frozenset(
    {
        # word processing
        ".doc", ".docx", ".docm", ".dotx", ".dotm", ".odt", ".rtf",
        # spreadsheets
        ".xls", ".xlsx", ".xlsm", ".xltx", ".ods",
        # presentations
        ".ppt", ".pptx", ".pptm", ".ppsx", ".potx", ".odp",
    }
)  # fmt: skip
RENDERABLE_SUFFIXES: frozenset[str] = (
    frozenset({".pdf"}) | TIFF_SUFFIXES | OFFICE_SUFFIXES
)
NATIVE_UNSUPPORTED_SUFFIXES: frozenset[str] = TIFF_SUFFIXES

USE_VLM_HINT = "Re-run with --conversion-method auto or vlm to OCR it via the VLM."


def _anydoc_extractor_label() -> str:
    """Front-matter `extractor` value for native output, e.g. 'anydoc v0.2.4'."""
    try:
        return f"anydoc v{package_version('firecrawl-anydoc')}"
    except PackageNotFoundError:
        return "anydoc (version unknown)"


ANYDOC_EXTRACTOR = _anydoc_extractor_label()


class NativeUnsupportedError(Exception):
    """Raised for an input format anydoc cannot read, before calling anydoc."""


@dataclass
class VlmResult:
    """Concatenated per-page Markdown plus the 1-indexed pages whose OCR failed."""

    markdown: str
    failed_pages: list[int]


def _is_retryable_ocr_error(exc: BaseException) -> bool:
    """Retry transient failures (timeouts, connection errors, 5xx). Do not
    retry 4xx client errors — retrying a malformed request won't fix it."""
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


async def ocr_page(
    client: httpx.AsyncClient,
    image_b64: str,
    *,
    base_url: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_retries: int,
) -> str:
    """Send a base64-encoded page image to a vLLM server and return the OCR text.

    Retries transient errors (timeouts, connection errors, 5xx) up to
    max_retries times with exponential backoff; raises immediately on 4xx.
    Raises the underlying exception once retries are exhausted — callers are
    responsible for catching it and converting it to a placeholder.

    Args:
        client: Shared httpx.AsyncClient, built once per run.
        image_b64: Base64-encoded PNG image string.
        base_url: Full URL of the vLLM server.
        model: Model identifier to request from the server.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        max_retries: Maximum attempts (including the first) before giving up.

    Returns:
        OCR text content from the model response.
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    }
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable_ocr_error),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    ):
        with attempt:
            response = await client.post(
                urljoin(base_url, "v1/chat/completions"), json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    raise AssertionError("unreachable: reraise=True re-raises the last error")


def convert_office_to_pdf(doc_path: Path) -> Path:
    """Convert an Office document to PDF via headless LibreOffice.

    Run synchronously due to LibreOffice blocking.

    Returns a PDF path inside a fresh temp dir.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="soffice-"))
    try:
        result = subprocess.run(
            [
                "soffice",
                "--headless",
                "--norestore",
                "--convert-to",
                "pdf",
                "--outdir",
                str(tmp_dir),
                str(doc_path),
            ],
            capture_output=True,
            timeout=DEFAULT_LIBREOFFICE_TIMEOUT_S,
            check=False,
        )
        pdf_path = tmp_dir / (doc_path.stem + ".pdf")
        if (
            result.returncode != 0
            or not pdf_path.exists()
            or pdf_path.stat().st_size == 0
        ):
            raise RuntimeError(
                f"LibreOffice conversion failed (exit {result.returncode}): "
                f"{result.stderr.decode(errors='replace')[:500]}"
            )
        return pdf_path
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def render_pages(
    input_file: Path,
    page_image_cache_dir: Path,
    file_hash: str,
    dpi: int = DEFAULT_CACHE_DPI,
) -> list[Path]:
    """Render every page/frame/slide of input_file to a full-resolution PNG.

    Writes to the shared cache convention
    page_image_cache_dir/<file_hash>/pNNN-MMM.png (required by
    resolve.py, which derives this same subdirectory name from
    the output .md file's own stem) and returns the ordered list of cache
    paths. Fully synchronous — see module docstring for why this stage
    doesn't need concurrency.
    """
    suffix = input_file.suffix.lower()
    if suffix == ".pdf":
        pages = convert_from_path(str(input_file), dpi=dpi)
    elif suffix in TIFF_SUFFIXES:
        # ImageSequence.Iterator re-seeks and yields the SAME underlying image
        # object for every frame (still tied to the open file handle); .copy()
        # forces each frame to materialize as an independent, decoded image
        # before the file handle it depends on potentially gets reused.
        pages = [
            frame.copy() for frame in ImageSequence.Iterator(Image.open(input_file))
        ]
    elif suffix in OFFICE_SUFFIXES:
        pdf_path = convert_office_to_pdf(input_file)
        try:
            pages = convert_from_path(str(pdf_path), dpi=dpi)
        finally:
            shutil.rmtree(pdf_path.parent, ignore_errors=True)
    else:
        raise ValueError(f"Unsupported file type: {input_file.suffix}")

    n_pages = len(pages)
    if n_pages == 0:
        raise ValueError(f"{input_file.name} has no pages/frames/slides to render")
    elif n_pages > 999_999:
        raise ValueError(
            f"{input_file.name} has {n_pages} pages/frames/slides, "
            "exceeds 999,999 limit for page-image cache naming"
        )
    cache_paths = []
    for i, page in enumerate(pages):
        cache_path = (
            page_image_cache_dir / file_hash / f"p{i + 1:06d}-{n_pages:06d}.png"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        page.save(cache_path, "PNG")
        cache_paths.append(cache_path)
    logger.debug(f"{input_file.name}: rendered and cached {n_pages} page(s)")
    return cache_paths


def _render_failure_hint(exc: BaseException) -> str:
    """Actionable suffix for render errors caused by a missing external tool."""
    if isinstance(exc, FileNotFoundError) and exc.filename == "soffice":
        return (
            " — LibreOffice (soffice) is not on PATH: run scripts/setup_libreoffice.sh,"
            " or use --conversion-method auto/native to extract Office files with anydoc."
            " Warning: extraction quality will be significantly worse when using anydoc."
        )
    if isinstance(exc, (PDFInfoNotInstalledError, PopplerNotInstalledError)):
        return (
            " — poppler (pdftoppm/pdfinfo) is not on PATH:"
            " conda install -c conda-forge poppler"
        )
    return ""


def resolve_conversion_method(input_file: Path, method_arg: str) -> str:
    """'auto' picks a per-format default: Office documents use anydoc
    ('native') since they're born-digital and anydoc extracts them fast and
    losslessly; PDFs and TIFFs use the VLM OCR pipeline ('vlm'), since
    they're scan-oriented with no reliable native text layer. An explicit
    --conversion-method vlm/native overrides the default for every file
    regardless of format."""
    if method_arg != "auto":
        return method_arg
    return "native" if input_file.suffix.lower() in OFFICE_SUFFIXES else "vlm"


async def convert_pages_vlm(
    page_paths: list[Path],
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    target_px: int,
    max_retries: int,
    source_name: str,
) -> VlmResult:
    """OCR an already-rendered, ordered list of page images and return
    concatenated Markdown plus the pages whose OCR failed.

    Format-agnostic: works identically for PDF pages, TIFF frames, and
    LibreOffice-rendered Office pages, since render_pages already reduced
    every input format to the same shape (numbered PNGs on disk). Each page
    section is prefixed with an HTML comment (<!-- page N -->); page order in
    the returned Markdown matches document order regardless of OCR completion
    order, since asyncio.gather preserves input order. Never raises — OCR
    failures (including retries exhausted inside ocr_page) become an
    <!-- OCR FAILED: ... --> placeholder for that page only and are listed
    in VlmResult.failed_pages.
    """

    async def _one_page(i: int, path: Path) -> tuple[str, bool]:
        marker = f"<!-- page {i + 1} -->"

        def _downscale_to_b64() -> str:
            img = Image.open(path).convert("RGB")
            img.thumbnail((target_px, target_px), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        image_b64 = await asyncio.to_thread(_downscale_to_b64)
        async with sem:
            try:
                text = await ocr_page(
                    client,
                    image_b64,
                    base_url=base_url,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    max_retries=max_retries,
                )
                return f"{marker}\n\n{text.strip()}", True
            except Exception as e:  # noqa: BLE001 - batch guard; exception types tallied by record_failure
                detail = record_failure("ocr", e)
                logger.warning(f"{source_name} p{i + 1}: OCR failed — {detail}")
                return f"{marker}\n\n<!-- OCR FAILED: {e} -->", False

    outcomes = await asyncio.gather(
        *(_one_page(i, p) for i, p in enumerate(page_paths))
    )
    return VlmResult(
        markdown="\n\n".join(section for section, _ in outcomes),
        failed_pages=[i + 1 for i, (_, ok) in enumerate(outcomes) if not ok],
    )


async def convert_native(input_file: Path) -> str:
    """Extract Markdown natively via anydoc, run directly against the
    original document.

    Anydoc returns one Markdown blob rather than a per-page split.

    Raises:
        NativeUnsupportedError: for formats anydoc cannot read (TIFF), without
            calling anydoc.
        anydoc.ConvertError: any anydoc failure, notably anydoc.NeedsOcrError
            for scanned PDFs. process_one_file turns these into explanations.
        OSError: the file could not be read.
    """
    suffix = input_file.suffix.lower()
    if suffix in NATIVE_UNSUPPORTED_SUFFIXES:
        raise NativeUnsupportedError(f"{suffix} files are not supported by anydoc")
    return await asyncio.to_thread(anydoc.to_markdown, str(input_file))


def _summarize_pages(pages: list[int]) -> str:
    """Compress sorted 1-indexed page numbers into ranges, e.g. '1-4, 7'."""
    ranges: list[tuple[int, int]] = []
    for page in pages:
        if ranges and page == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], page)
        else:
            ranges.append((page, page))
    return ", ".join(
        str(first) if first == last else f"{first}-{last}" for first, last in ranges
    )


def explain_native_failure(exc: Exception) -> tuple[str, str]:
    """Map a native-extraction error to (failure-tally stage, user-facing reason)."""
    if isinstance(exc, anydoc.NeedsOcrError):
        pages: list[int] = getattr(exc, "pages", [])
        where = (
            f"page(s) {_summarize_pages(pages)} of {exc.page_count} have"
            if pages
            else f"pages ({exc}) have"
        )
        return (
            "anydoc_needs_ocr",
            (
                f"scanned PDF — {where} no text layer and need OCR, which anydoc "
                f"does not do. {USE_VLM_HINT}"
            ),
        )
    if isinstance(exc, (NativeUnsupportedError, anydoc.UnsupportedError)):
        return (
            "anydoc_unsupported",
            f"not convertible by anydoc ({exc}). {USE_VLM_HINT}",
        )
    if isinstance(exc, anydoc.EncryptedError):
        return (
            "anydoc_encrypted",
            "document is encrypted or password-protected; remove the protection and re-run.",
        )
    return "anydoc", f"anydoc extraction failed ({type(exc).__name__}: {exc})."


async def process_one_file(
    input_file: Path,
    file_hash: str,
    page_paths: list[Path],
    output_dir: Path,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    args: argparse.Namespace,
    *,
    method: str,
) -> dict[str, str]:
    """Convert one file to Markdown with its resolved method and write the output.

    Returns {"status": "converted" | "partial" | "failed", "name": ...}.
    "partial" indicates some VLM pages failed OCR.
    "failed" indicates that no .md was written.
    """
    start = time.perf_counter()
    failed = {"status": "failed", "name": input_file.name}
    try:
        if method == "vlm":
            vlm_result = await convert_pages_vlm(
                page_paths,
                client,
                sem,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                target_px=args.target_px,
                max_retries=args.max_retries,
                source_name=input_file.name,
            )
            if len(vlm_result.failed_pages) == len(page_paths):
                reason = (
                    f"OCR failed on all {len(page_paths)} page(s)"
                    if page_paths
                    else "no pages were rendered"
                )
                logger.error(
                    f"{input_file.name}: {reason}; no output written "
                    "(it will be retried on the next run)."
                )
                return failed
            markdown = vlm_result.markdown
            front_matter = build_front_matter(
                file_hash=file_hash,
                raw_file_path=str(input_file),
                conversion_method=method,
                extractor=args.model,
                ocr_failed_pages=vlm_result.failed_pages or None,
            )
            status = "partial" if vlm_result.failed_pages else "converted"
        else:
            try:
                markdown = await convert_native(input_file)
            except (anydoc.ConvertError, NativeUnsupportedError) as e:
                stage, reason = explain_native_failure(e)
                record_failure(stage, e)
                logger.error(f"{input_file.name}: {reason}")
                return failed
            front_matter = build_front_matter(
                file_hash=file_hash,
                raw_file_path=str(input_file),
                conversion_method=method,
                extractor=ANYDOC_EXTRACTOR,
            )
            status = "converted"

        output_path = output_dir / (file_hash + ".md")
        await asyncio.to_thread(
            output_path.write_text, front_matter + markdown, encoding="utf-8"
        )
        elapsed = time.perf_counter() - start
        if status == "partial":
            logger.warning(
                f"{input_file.name}: OCR failed on page(s) "
                f"{_summarize_pages(vlm_result.failed_pages)} of {len(page_paths)}; "
                "wrote partial output (see ocr_failed_pages). Use --force to retry."
            )
        logger.debug(f"Converted {input_file.name} ({method}) in {elapsed:.1f}s")
        return {"status": status, "name": input_file.name}
    except Exception as e:  # noqa: BLE001 - batch guard; exception types tallied by record_failure
        elapsed = time.perf_counter() - start
        detail = record_failure("convert_file", e)
        logger.error(f"{input_file.name}: failed after {elapsed:.1f}s — {detail}")
        return failed


async def run_async(
    files_to_process: list[Path],
    file_hash_by_file: dict[Path, str],
    page_paths_by_file: dict[Path, list[Path]],
    method_by_file: dict[Path, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    """Convert all files concurrently with their resolved methods.

    One coroutine per file (outer, progress-tracked via atqdm.gather); within
    each "vlm" file, one coroutine per page. "native" files have no entry in
    page_paths_by_file.
    A single shared semaphore bounds total in-flight OCR requests across every
    file and page combined.
    """
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=DEFAULT_REQUEST_TIMEOUT_S) as client:
        return await atqdm.gather(
            *(
                process_one_file(
                    f,
                    file_hash_by_file[f],
                    page_paths_by_file.get(f, []),
                    output_dir,
                    client,
                    sem,
                    args,
                    method=method_by_file[f],
                )
                for f in files_to_process
            ),
            desc="Converting files",
            unit="file",
        )


def aggregate_stats(results: list[dict[str, str]]) -> dict[str, int]:
    """Tally converted/partial/failed counts from per-file result dicts."""
    stats = {"converted": 0, "partial": 0, "failed": 0}
    for r in results:
        stats[r["status"]] += 1
    return stats


def _needs_ocr_check(input_files: list[Path], conversion_method_arg: str) -> bool:
    """True if any file in the batch will be processed by the OCR model, e.g.:
        - PDFs/TIFFs (unless `--conversion-method='native'` is specified);
        - Office files when `--conversion-method='vlm'` is specified.

    A native-only batch of .docx/.pptx does not require VLM OCR.
    """
    return any(
        resolve_conversion_method(f, conversion_method_arg) == "vlm"
        for f in input_files
    )


def _warn_native_unsupported(method_by_file: dict[Path, str]) -> None:
    """Flag inputs for which `--conversion-method='native'` is guaranteed to fail."""
    doomed = [
        f.name
        for f, method in method_by_file.items()
        if method == "native" and f.suffix.lower() in NATIVE_UNSUPPORTED_SUFFIXES
    ]
    if doomed:
        logger.warning(
            f"{len(doomed)} file(s) can't be converted natively (anydoc has no TIFF "
            f"support) and will fail: {', '.join(doomed)}. "
            "Use --conversion-method auto or vlm for them."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDFs, TIFFs, and MS Office / OpenDocument files to Markdown (async).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing input PDF/TIFF/Office files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory for output Markdown files.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="vLLM URL (e.g. http://localhost:8002) to which requests will be sent.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-process files that already have a Markdown output.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--page-image-cache",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to write rendered page images for debugging. Defaults to <output-dir>/page_image_cache.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum concurrent OCR requests to the vLLM server, shared across all files.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries for transient OCR failures (timeouts, connection errors, 5xx).",
    )
    parser.add_argument(
        "--conversion-method",
        choices=CONVERSION_METHODS,
        default=DEFAULT_CONVERSION_METHOD,
        help="How to extract Markdown. 'auto': VLM OCR for PDF/TIFF, anydoc for "
        "Office/OpenDocument files. 'vlm' (recommended): OCR every file via the vLLM server "
        "(Office files are rendered through LibreOffice first). 'native': extract "
        "every file with anydoc (no OCR, no page markers); scanned PDFs and TIFFs "
        "fail with an explanation.",
    )

    gen_group = parser.add_argument_group(
        "generation",
        "Arguments that determine output content and are included in the version hash.",
    )
    gen_group.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help="Model identifier to request from the vLLM server.",
    )
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate per page.",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability.",
    )
    gen_group.add_argument(
        "--target-px",
        type=int,
        default=1540,
        help="Downscale the OCR-bound image to this longest-dimension cap on the 'vlm' path. "
        f"Pages are still cached at the full {DEFAULT_CACHE_DPI} DPI render regardless "
        "of this value.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level="DEBUG" if args.verbose else "INFO",
        colorize=True,
    )

    if not args.input_dir.is_dir():
        logger.error(f"Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    input_files = sorted(
        f
        for f in args.input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in RENDERABLE_SUFFIXES
    )
    if not input_files:
        logger.warning(f"No PDF/TIFF/Office files found in '{args.input_dir}'.")
        sys.exit(0)

    output_dir = args.output_dir
    logger.info(f"Model: {args.model} (conversion-method={args.conversion_method})")
    logger.info(f"{args.input_dir} → {output_dir} ({len(input_files)} files)")

    stats = {"total": 0, "skipped": 0}
    files_to_process: list[Path] = []
    file_hash_by_file: dict[Path, str] = {}
    for input_file in input_files:
        stats["total"] += 1
        file_hash = compute_file_hash(input_file)
        file_hash_by_file[input_file] = file_hash
        output_path = output_dir / (file_hash + ".md")
        if output_path.exists() and not args.force:
            logger.debug(
                f"Skipping {input_file.name} — output exists. Use --force to overwrite."
            )
            stats["skipped"] += 1
            continue
        files_to_process.append(input_file)

    # resolve each file's backend
    method_by_file = {
        f: resolve_conversion_method(f, args.conversion_method)
        for f in files_to_process
    }
    _warn_native_unsupported(method_by_file)

    if _needs_ocr_check(files_to_process, args.conversion_method):
        check_model_available(args.base_url, args.model)
    else:
        logger.info(
            "No OCR-bound files in this batch — skipping vLLM availability check."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # write a copy of all command-line arguments to the output directory for reproducibility
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        f.write(serialize_args(args))
        logger.debug(f"Wrote config to {output_dir / 'config.json'}")

    if args.page_image_cache is None:
        args.page_image_cache = output_dir / "page_image_cache"
    args.page_image_cache.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Stage 1: render pages to cached images for every file routed to "vlm".
    files_to_render = [f for f in files_to_process if method_by_file[f] == "vlm"]
    page_paths_by_file: dict[Path, list[Path]] = {}
    render_failures: list[dict[str, str]] = []
    for input_file in tqdm(files_to_render, desc="Rendering pages", unit="file"):
        try:
            page_paths_by_file[input_file] = render_pages(
                input_file, args.page_image_cache, file_hash_by_file[input_file]
            )
        except Exception as e:  # noqa: BLE001 - batch guard; exception types tallied by record_failure
            detail = record_failure("render", e)
            logger.error(
                f"{input_file.name}: rendering failed — {detail}{_render_failure_hint(e)}"
            )
            render_failures.append({"status": "failed", "name": input_file.name})

    t1 = time.perf_counter()
    if files_to_render:
        logger.info(
            f"Page rendering completed for {len(files_to_render)} file(s) in "
            f"{t1 - t0:.1f}s"
        )

    t2 = time.perf_counter()
    # Stage 2: convert every native file and every successfully-rendered vlm file.
    files_to_convert = [
        f
        for f in files_to_process
        if method_by_file[f] == "native" or f in page_paths_by_file
    ]

    convert_results = asyncio.run(
        run_async(
            files_to_convert,
            file_hash_by_file,
            page_paths_by_file,
            method_by_file,
            output_dir,
            args,
        )
    )
    t2 = time.perf_counter()
    logger.info(
        f"Conversion completed for {len(files_to_convert)} file(s) in {t2 - t1:.1f}s"
    )

    stats.update(aggregate_stats(render_failures + convert_results))
    d = t2 - t0
    logger.info(
        f"Done in {d:.1f}s — "
        f"{stats['total']} total, {stats['converted']} converted, "
        f"{stats['partial']} partial, {stats['skipped']} skipped, "
        f"{stats['failed']} failed."
    )
    log_failure_summary()


if __name__ == "__main__":
    main()
