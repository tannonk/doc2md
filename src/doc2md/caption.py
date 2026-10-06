"""Generate captions for images and insert them as markdown alt text + note blocks.

Reads markdown files produced by resolve.py, where resolved
images appear as empty-alt-text references like:

    ![](images/mikrogen_control_sera_p002_image_4.png)

For each such reference, an image captioning model is called and two
distinct outputs are inserted: a brief description in the alt text, and a
longer, type-tagged caption in a blockquote note directly below:

    ![Photograph of a reagent vial with a printed lot number label.](images/mikrogen_control_sera_p002_image_4.png)

    > [photograph] Photograph of a labeled reagent vial (mikrogen control
    > serum) showing lot number and expiry date printed on the cap.

Leading indentation is preserved, so refs nested inside list items stay
correctly nested.

Unresolved leftover placeholders from earlier stages (e.g. `![image](image_1.png)`,
which have no `images/` prefix and point to no real file) are left untouched.

Captioning uses LiteLLM as a generalizable multi-provider interface, so the
same script works against a local vLLM server (the default) or any hosted
provider LiteLLM supports (Anthropic, OpenAI, ...) by changing --model alone.
--model takes a bare name (e.g. "nvidia/Nemotron-..."); the script asks the
server at --base-url what it has loaded and prefixes the name with
hosted_vllm/ itself when it matches, falling back to the name as given (a
LiteLLM provider string) otherwise. The hosted_vllm/ prefix may still be
given explicitly. Structured output is enforced via Instructor. Requests
run concurrently under an asyncio.Semaphore.

Output is written to a new `captioned/` directory next to the input, mirroring
resolve.py's own `resolved/` output convention. Images are not
copied. `captioned/images` is a relative symlink back to the source `images/`
directory.

Note: a file is only reprocessed with --force if `captioned/<name>.md` already
exists. A run with some failed captions (see failed_captions in the summary)
is not automatically retried on the next invocation — rerun with --force.

Example usage:
    
    # start a local vLLM server for 
    # caption generation (must have image input support)
    bash docker/nemotron-3-omni/serve.sh
    
    # run captioning against that server
    python -m doc2md.caption \
        -i <path-to-dir-containing-resolved-markdown>
    
    python -m doc2md.caption \
        -i data/examples/md_vlm/resolved
"""

import argparse
import asyncio
import re
import shutil
import sys
import time
from enum import Enum
from pathlib import Path

import instructor
import litellm
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from doc2md.utils import (
    image_to_data_uri,
    list_served_models,
    log_failure_summary,
    record_failure,
)

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"

IMAGE_REF_PATTERN = re.compile(
    r"^([ \t]*)!\[\]\(images/([^()\n]+\.png)\)$", re.MULTILINE
)

CAPTION_PROMPT = (
    "Classify and describe this figure for a search index. Use `pictogram` "
    "specifically for GHS/hazard warning symbols, distinct from `logo` or `other`."
)


class ImageType(str, Enum):
    LOGO = "logo"
    PICTOGRAM = "pictogram"
    DIAGRAM = "diagram"
    GRAPH = "graph"
    TABLE = "table"
    PHOTOGRAPH = "photograph"
    SCREENSHOT = "screenshot"
    OTHER = "other"


class ImageDescription(BaseModel):
    image_type: ImageType = Field(description="The category of figure this is.")
    alt_text: str = Field(
        description="Brief (under ~15 words) markdown alt-text description. "
        "State the image type and what it shows."
    )
    caption: str = Field(
        description="Longer, more detailed caption (2-4 sentences) for a note "
        "below the image. State the image type, what it depicts, any "
        "axis/legend/hazard-code labels, and the key takeaway."
    )


def sanitize_text(text: str, max_len: int, strip_brackets: bool = False) -> str:
    """Collapse whitespace/newlines, optionally strip brackets that would break
    `![<alt>](path)` alt-text syntax, then cap length."""
    collapsed = " ".join(text.split())
    if strip_brackets:
        collapsed = collapsed.replace("[", "(").replace("]", ")")
    return collapsed[:max_len]


def rewrite_markdown(
    text: str, descriptions: dict[str, ImageDescription | None]
) -> tuple[str, int, int]:
    """Insert alt text and a captioned blockquote note below each image reference.

    Refs with a missing/failed description are left byte-identical (empty alt,
    no note block).

    Args:
        text: Markdown content containing zero or more image references.
        descriptions: Map of image filename -> ImageDescription, or None if
            captioning failed.

    Returns:
        (updated_text, n_captioned, n_failed).
    """
    n_captioned = 0
    n_failed = 0

    def replace(m: re.Match) -> str:
        nonlocal n_captioned, n_failed
        indent, name = m.group(1), m.group(2)
        desc = descriptions.get(name)
        if desc:
            n_captioned += 1
            alt = sanitize_text(desc.alt_text, max_len=200, strip_brackets=True)
            caption = sanitize_text(desc.caption, max_len=600)
            return f"{indent}![{alt}](images/{name})\n\n{indent}> [{desc.image_type.value}] {caption}"
        n_failed += 1
        return m.group(0)

    updated = IMAGE_REF_PATTERN.sub(replace, text)
    return updated, n_captioned, n_failed


def _is_local_server(model: str) -> bool:
    """Return True if model targets a self-hosted OpenAI-compatible server (e.g. vLLM).

    Only hosted_vllm/ counts: a bare openai/ model string is ambiguous (it's
    also LiteLLM's hosted OpenAI provider) and is resolved by
    resolve_litellm_model before this is ever checked — see its docstring.
    """
    return model.startswith("hosted_vllm/")


def resolve_litellm_model(model: str, base_url: str) -> str:
    """Resolve a bare model name to the LiteLLM identifier that reaches it.

    --model accepts either a bare Hugging Face-style name (e.g.
    "nvidia/Nemotron-...") or an explicit LiteLLM provider string (e.g.
    "hosted_vllm/nvidia/Nemotron-...", "anthropic/claude-..."). A bare name
    is ambiguous — it may be a model loaded on the self-hosted server at
    base_url, or a hosted-provider model whose org happens to collide with a
    LiteLLM provider name (e.g. "openai/gpt-oss-20b" is both an OpenAI-format
    HF repo and LiteLLM's hosted "openai" provider) — so this asks the server
    what it actually has loaded rather than guessing from the string alone.

    Resolution order:
        1. Already hosted_vllm/-prefixed: kept as is, but still checked
           against the server so an unavailable model fails fast.
        2. Served by the self-hosted server at base_url: prefixed with
           hosted_vllm/ (the local server wins over a same-named hosted
           provider).
        3. A LiteLLM-recognized provider string otherwise (e.g.
           "anthropic/claude-3-5-sonnet"): returned unchanged. The server is
           not required to be reachable in this case.
        4. Otherwise: raises, since the model can be reached neither
           locally nor through a known LiteLLM provider.

    Args:
        model: The --model value as given on the command line.
        base_url: Base URL of the self-hosted OpenAI-compatible server.

    Returns:
        The LiteLLM model identifier to pass to litellm.acompletion.

    Raises:
        RuntimeError: The model is not served locally and not a recognized
            LiteLLM provider string.
    """
    if _is_local_server(model):
        served = list_served_models(base_url)
        local_name = model.split("/", 1)[1]
        if local_name not in served:
            raise RuntimeError(
                f"Model '{local_name}' not found on vLLM server at {base_url}. "
                f"Available models: {served}"
            )
        return model

    try:
        served = list_served_models(base_url)
        server_error = None
    except RuntimeError as e:
        served = []
        server_error = e

    if model in served:
        return f"hosted_vllm/{model}"

    try:
        litellm.get_llm_provider(model)
    except Exception as e:
        reachability = (
            f"not served at {base_url} (available: {served})"
            if server_error is None
            else f"server at {base_url} unreachable ({server_error})"
        )
        raise RuntimeError(
            f"Model '{model}' is {reachability} and is not a LiteLLM "
            f"provider string ({e})."
        ) from e

    return model


def _litellm_api_base(base_url: str) -> str:
    """Append /v1 for litellm's hosted_vllm/openai providers.

    litellm's get_complete_url() just appends "chat/completions" to api_base —
    it does not add /v1 itself — so api_base must already include it. --base-url
    stays a bare server root (matching convert.py's urljoin convention);
    this is where that gets reconciled with litellm's expectation.
    """
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def build_caption_client() -> instructor.AsyncInstructor:
    """Return an Instructor client wrapping litellm.acompletion.

    Call surface is client.create(...), not client.chat.completions.create(...) —
    from_litellm wraps a bare function, so there is no chat.completions namespace.

    Uses JSON mode rather than the tool-calling default: captioning VLMs like
    LightOnOCR aren't guaranteed to support OpenAI-style function/tool calling,
    and vLLM only accepts tool_choice when the server was launched with a
    matching --tool-call-parser. JSON mode works against plain chat completion.
    """
    return instructor.from_litellm(
        litellm.acompletion, mode=instructor.Mode.JSON, async_client=True
    )


async def caption_one_image(
    client: instructor.AsyncInstructor,
    image_path: Path,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    max_tokens: int,
    temperature: float,
    max_image_dimension: int,
    max_retries: int,
    sem: asyncio.Semaphore,
) -> ImageDescription | None:
    """Caption a single image. Returns None (and logs a warning) on any failure."""
    async with sem:
        try:
            data_uri = await asyncio.to_thread(
                lambda: image_to_data_uri(Image.open(image_path), max_image_dimension)
            )
            kwargs: dict = {
                "model": model,
                "response_model": ImageDescription,
                "max_retries": max_retries,
                "num_retries": max_retries,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CAPTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
            }
            if _is_local_server(model):
                kwargs["api_base"] = _litellm_api_base(base_url)
                if api_key:
                    kwargs["api_key"] = api_key
            return await client.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - batch guard; exception types tallied by record_failure
            logger.warning(
                f"{image_path.name}: caption failed — {record_failure('caption', e)}"
            )
            return None


async def process_file(
    client: instructor.AsyncInstructor,
    md_path: Path,
    images_dir: Path,
    out_path: Path,
    sem: asyncio.Semaphore,
    args: argparse.Namespace,
) -> dict[str, int]:
    """Caption every image referenced in one markdown file and write the result.

    Returns a dict of per-file counters: images, captioned, failed.
    """
    text = md_path.read_text(encoding="utf-8")
    # dict.fromkeys dedupes while preserving first-seen order (same image may repeat).
    names = list(dict.fromkeys(m.group(2) for m in IMAGE_REF_PATTERN.finditer(text)))

    if not names:
        out_path.write_text(text, encoding="utf-8")
        return {"images": 0, "captioned": 0, "failed": 0}

    results = await asyncio.gather(
        *(
            caption_one_image(
                client,
                images_dir / name,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_image_dimension=args.max_image_dimension,
                max_retries=args.max_retries,
                sem=sem,
            )
            for name in names
        )
    )
    descriptions = {name: result for name, result in zip(names, results)}
    updated, n_ok, n_fail = rewrite_markdown(text, descriptions)
    out_path.write_text(updated, encoding="utf-8")
    return {"images": len(names), "captioned": n_ok, "failed": n_fail}


async def run_async(
    files_to_process: list[Path],
    images_dir: Path,
    captioned_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, int]]:
    """Caption all files concurrently, gated by a shared semaphore, streaming
    each file's output to disk as soon as it finishes."""
    client = build_caption_client()
    sem = asyncio.Semaphore(args.concurrency)
    return await atqdm.gather(
        *(
            process_file(client, f, images_dir, captioned_dir / f.name, sem, args)
            for f in files_to_process
        ),
        desc="Captioning files",
        unit="file",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate image captions and insert them as markdown alt text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory of resolved markdown files produced by resolve.py.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of a self-hosted OpenAI-compatible server (e.g. vLLM), "
        "queried to resolve a bare --model name. Ignored once --model is an "
        "explicit non-hosted_vllm/ LiteLLM provider string.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help="Model to caption with. A bare name (e.g. nvidia/Nemotron-...) is "
        "checked against --base-url and prefixed with hosted_vllm/ "
        "automatically if served there; otherwise it's treated as a "
        "LiteLLM provider string (e.g. anthropic/claude-...). "
        "hosted_vllm/<model> may also be given explicitly.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key to forward to the model provider. Not needed for an "
        "unauthenticated local vLLM server.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum concurrent captioning requests.",
    )
    parser.add_argument(
        "--max-image-dimension",
        type=int,
        default=1024,
        help="Downscale images so their longest edge fits within this many pixels "
        "before sending to the captioning model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="Maximum tokens to generate per caption.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries for both structured-output validation and transport failures.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-process files that already have a captioned output.",
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

    if not args.input_dir.is_dir():
        logger.error(f"Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    resolved_model = resolve_litellm_model(args.model, args.base_url)
    if resolved_model != args.model:
        logger.debug(f"Resolved model '{args.model}' -> '{resolved_model}'.")
    args.model = resolved_model

    images_dir = args.input_dir / "images"
    if not images_dir.is_dir():
        logger.error(f"Images directory '{images_dir}' does not exist.")
        sys.exit(1)

    captioned_dir = args.input_dir / "captioned"
    # cp -r images_dir to captioned_dir/images
    shutil.copytree(images_dir, captioned_dir / "images", dirs_exist_ok=True)

    md_files = sorted(args.input_dir.glob("*.md"))
    if not md_files:
        logger.warning(f"No markdown files found in '{args.input_dir}'.")
        sys.exit(0)

    stats = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "images_captioned": 0,
        "failed_captions": 0,
    }
    files_to_process: list[Path] = []
    for md_path in md_files:
        stats["total"] += 1
        output_path = captioned_dir / md_path.name
        if output_path.exists() and not args.force:
            logger.debug(
                f"Skipping {md_path.name} — captioned output already exists. Use --force to overwrite."
            )
            stats["skipped"] += 1
            continue
        files_to_process.append(md_path)

    logger.info(f"Model: {args.model}")
    logger.info(f"{args.input_dir} → {captioned_dir} ({len(files_to_process)} files)")

    wall_start = time.perf_counter()
    results = asyncio.run(run_async(files_to_process, images_dir, captioned_dir, args))
    total_elapsed = time.perf_counter() - wall_start

    for r in results:
        stats["processed"] += 1
        stats["images_captioned"] += r["captioned"]
        stats["failed_captions"] += r["failed"]

    logger.info(
        f"Done in {total_elapsed:.1f}s — "
        f"{stats['total']} total files, {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['images_captioned']} images captioned, "
        f"{stats['failed_captions']} failed."
    )
    logger.info(f"Captioned files written to: {captioned_dir}")
    log_failure_summary()


if __name__ == "__main__":
    main()
