"""Resolve bbox image placeholders in LightOnOCR-2 bbox-model markdown output.

Reads markdown files produced by convert.py using a bbox-capable model,
crops detected regions from cached page images, and writes them as PNG files next
to the markdown. Each bbox placeholder like:

    ![image](image_1.png)670,83,932,167

is replaced with a relative markdown image reference:

    ![image](images/doc_p001_image_1.png)

Requires page images saved by convert.py --page-image-cache. 

Pass --embed-base64 to inline cropped images as data URIs instead of separate files
at the cost of significantly larger markdown files.

Note: for image placeholder that do not have bbox coordinates (e.g., ![image](image_1.png)), 
the script will leave them unchanged.

Example usage:

    # Resolve bbox images and save them as PNG files next to the markdown:
    python -m doc2md.resolve \
        -i <path-to-markdown-dir>
"""

import argparse
import re
import sys
import time
from pathlib import Path

from loguru import logger
from PIL import Image
from tqdm import tqdm

from doc2md.utils import (
    image_to_data_uri,
    log_failure_summary,
    record_failure,
)

BBOX_PATTERN = re.compile(r"!\[image\]\((image_\d+\.png)\)\s*(\d+),(\d+),(\d+),(\d+)")
PAGE_MARKER_PATTERN = re.compile(r"<!-- page (\d+) -->")


def crop_from_bbox(
    source_image: Image.Image,
    coords: tuple[int, int, int, int],
    padding: int = 20,
) -> Image.Image:
    """Crop a region from source_image using normalized [0, 1000] coordinates."""
    w, h = source_image.size
    x1, y1, x2, y2 = coords
    px1 = max(0, int(x1 * w / 1000) - padding)
    py1 = max(0, int(y1 * h / 1000) - padding)
    px2 = min(w, int(x2 * w / 1000) + padding)
    py2 = min(h, int(y2 * h / 1000) + padding)
    logger.debug(
        f"Cropping bbox {coords} -> pixels ({px1},{py1},{px2},{py2}) from image size {w}x{h}"
    )
    return source_image.crop((px1, py1, px2, py2))


def find_cached_page(
    page_image_cache_dir: Path, pdf_stem: str, page_num: int
) -> Path | None:
    """Locate a cached page PNG using a glob on the total-pages suffix.

    convert.py saves images as p{N:06d}-{total:06d}.png; the total
    is not known at post-process time, so we glob for the page number prefix.
    """
    matches = list((page_image_cache_dir / pdf_stem).glob(f"p{page_num:06d}-*.png"))
    return matches[0] if matches else None


def resolve_section(
    section_text: str,
    page_num: int,
    pdf_stem: str,
    page_image_cache_dir: Path,
    output_images_dir: Path | None,
    embed_base64: bool,
    padding: int = 40,
) -> tuple[str, int, int]:
    """Replace all bbox placeholders in one page section with resolved images.

    Returns (updated_text, n_resolved, n_failed).
    """
    if not BBOX_PATTERN.search(section_text):
        return section_text, 0, 0

    cached_path = find_cached_page(page_image_cache_dir, pdf_stem, page_num)
    if cached_path is None:
        logger.warning(
            f"{pdf_stem} p{page_num}: no cached page image — leaving placeholders unchanged"
        )
        n_placeholders = len(BBOX_PATTERN.findall(section_text))
        return section_text, 0, n_placeholders

    source_image = Image.open(cached_path).convert("RGB")
    n_resolved = 0
    n_failed = 0

    def replace_match(m: re.Match) -> str:
        nonlocal n_resolved, n_failed
        ref = m.group(1)
        coords = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        try:
            cropped = crop_from_bbox(source_image, coords, padding=padding)
            if embed_base64:
                uri = image_to_data_uri(cropped)
                n_resolved += 1
                return f"![]({uri})"
            else:
                ref_base = ref.removesuffix(".png")
                out_name = f"{pdf_stem}_p{page_num:03d}_{ref_base}.png"
                cropped.save(output_images_dir / out_name, format="PNG")
                n_resolved += 1
                return f"![](images/{out_name})"
        except Exception as e:  # noqa: BLE001 - batch guard; exception types tallied by record_failure
            logger.warning(
                f"{pdf_stem} p{page_num} {ref}: crop failed — {record_failure('crop', e)}"
            )
            n_failed += 1
            return m.group(0)

    updated = BBOX_PATTERN.sub(replace_match, section_text)
    return updated, n_resolved, n_failed


def resolve_markdown(
    md_path: Path,
    page_image_cache_dir: Path,
    embed_base64: bool,
    padding: int = 40,
    output_dir: Path | None = None,
) -> tuple[str, int, int]:
    """Resolve all bbox placeholders in a markdown file.

    Splits the file on <!-- page N --> markers so each section is processed
    against the correct cached page image. Returns (updated_markdown,
    total_resolved, total_failed).

    Args:
        output_dir: Directory where resolved images are written. Defaults to
            the source file's parent, but callers should pass the intended
            output directory so image paths in the markdown stay consistent.
    """
    text = md_path.read_text(encoding="utf-8")
    pdf_stem = md_path.stem
    effective_output_dir = output_dir if output_dir is not None else md_path.parent

    if not embed_base64:
        images_dir = effective_output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
    else:
        images_dir = None

    # Zero-width lookahead splits before each marker without consuming it,
    # so every resulting section still starts with its own <!-- page N --> tag.
    sections = re.split(r"(?=<!-- page \d+ -->)", text)

    resolved_sections: list[str] = []
    total_resolved = total_failed = 0

    for section in sections:
        m = PAGE_MARKER_PATTERN.match(section)
        if m:
            page_num = int(m.group(1))
            updated, n_res, n_fail = resolve_section(
                section,
                page_num,
                pdf_stem,
                page_image_cache_dir,
                images_dir,
                embed_base64,
                padding,
            )
            resolved_sections.append(updated)
            total_resolved += n_res
            total_failed += n_fail
        else:
            resolved_sections.append(section)

    return "".join(resolved_sections), total_resolved, total_failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve bbox image placeholders in LightOnOCR-2 markdown output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory of raw markdown files produced by convert.py.",
    )
    parser.add_argument(
        "--page-image-cache",
        type=Path,
        required=False,
        metavar="DIR",
        help="Directory of cached page images from convert.py --page-image-cache. If not provided, the script will attempt to locate the cache in the same parent directory as --input-dir.",
    )
    parser.add_argument(
        "--embed-base64",
        action="store_true",
        help="Inline cropped images as base64 data URIs instead of saving PNG files.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite already-resolved output files.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=100,
        metavar="PX",
        help="Pixels of padding added on each side of the cropped region (rendered image pixels).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level="DEBUG" if args.verbose else "INFO",
        colorize=True,
    )

    # validate input directory
    if not args.input_dir.is_dir():
        logger.error(f"Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    # resolve image cache directory
    if args.page_image_cache is None:
        # infer the image cache directory from the input directory if not explicitly provided
        inferred_cache = args.input_dir / "page_image_cache"
        if inferred_cache.is_dir():
            args.page_image_cache = inferred_cache
            logger.info(f"Inferred image cache directory: {args.page_image_cache}")
        else:
            logger.error(
                f"No image cache directory provided and {inferred_cache} does not exist. Please specify --page-image-cache."
            )
            sys.exit(1)

    # gather all markdown files in the input directory
    md_files = sorted(args.input_dir.glob("*.md"))
    if not md_files:
        logger.warning(f"No markdown files found in '{args.input_dir}'.")
        sys.exit(0)

    # create output directory for resolved output files
    resolved_dir = args.input_dir / "resolved"
    resolved_dir.mkdir(exist_ok=True)

    logger.info(
        f"{args.input_dir} → {resolved_dir} ({len(md_files)} files) | cache: {args.page_image_cache}"
    )

    stats = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "images_resolved": 0,
        "failures": 0,
    }
    wall_start = time.perf_counter()

    for md_path in tqdm(md_files, desc="Resolving bbox images", unit="file"):
        stats["total"] += 1
        output_path = resolved_dir / md_path.name

        if not args.force and output_path.exists():
            logger.debug(
                f"Skipping {md_path.name} — resolved output already exists. Use --force to overwrite."
            )
            stats["skipped"] += 1
            continue

        start = time.perf_counter()
        try:
            updated, n_resolved, n_failed = resolve_markdown(
                md_path,
                args.page_image_cache,
                args.embed_base64,
                args.padding,
                output_dir=resolved_dir,
            )
            output_path.write_text(updated, encoding="utf-8")
            elapsed = time.perf_counter() - start
            logger.debug(
                f"{md_path.name}: {n_resolved} resolved, {n_failed} failed in {elapsed:.1f}s"
            )
            stats["processed"] += 1
            stats["images_resolved"] += n_resolved
            stats["failures"] += n_failed
        except Exception as e:  # noqa: BLE001 - batch guard; exception types tallied by record_failure
            logger.error(
                f"{md_path.name}: failed — {record_failure('resolve_file', e)}"
            )
            stats["failures"] += 1

    total_elapsed = time.perf_counter() - wall_start
    logger.info(
        f"Done in {total_elapsed:.1f}s — "
        f"{stats['total']} files, {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['images_resolved']} images resolved, "
        f"{stats['failures']} failures."
    )
    logger.info(f"Resolved files written to: {resolved_dir}")
    log_failure_summary()


if __name__ == "__main__":
    main()
