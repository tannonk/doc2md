"""Offline tests for the image-captioning stage.

Covers the pure regex/rewrite logic and the async per-file orchestration
against a duck-typed fake captioning client.
"""

import asyncio

import pytest
from PIL import Image

from doc2md import caption as caption_mod
from doc2md.caption import (
    IMAGE_REF_PATTERN,
    ImageDescription,
    ImageType,
    _is_local_server,
    process_file,
    resolve_litellm_model,
    rewrite_markdown,
    sanitize_text,
)


def test_image_ref_pattern_matches_resolved_refs():
    text = "before\n![](images/doc_p001_image_1.png)\nafter"
    matches = IMAGE_REF_PATTERN.findall(text)
    assert matches == [("", "doc_p001_image_1.png")]


def test_image_ref_pattern_captures_leading_indentation():
    text = "- Atemschutz:\n\n  ![](images/doc_p001_image_1.png)\n\n  text"
    matches = IMAGE_REF_PATTERN.findall(text)
    assert matches == [("  ", "doc_p001_image_1.png")]


def test_image_ref_pattern_ignores_unresolved_placeholders():
    text = "table cell ![image](image_1.png) icon"
    assert IMAGE_REF_PATTERN.findall(text) == []


def test_image_ref_pattern_ignores_already_captioned_refs():
    text = "![a photo of a vial](images/doc_p001_image_1.png)"
    assert IMAGE_REF_PATTERN.findall(text) == []


def test_sanitize_text_collapses_whitespace():
    assert (
        sanitize_text("  a\ncaption  with words  ", max_len=500)
        == "a caption with words"
    )


def test_sanitize_text_strips_brackets_only_when_requested():
    assert (
        sanitize_text("a [bracketed] value", max_len=500, strip_brackets=True)
        == "a (bracketed) value"
    )
    assert (
        sanitize_text("a [bracketed] value", max_len=500, strip_brackets=False)
        == "a [bracketed] value"
    )


def test_sanitize_text_caps_length():
    assert sanitize_text("x" * 10, max_len=3) == "xxx"


def test_rewrite_markdown_inserts_alt_text_and_note_block():
    text = "![](images/a.png)\nsome prose\n![](images/b.png)\n![image](image_1.png)\n"
    descriptions = {
        "a.png": ImageDescription(
            image_type=ImageType.PHOTOGRAPH,
            alt_text="A red vial.",
            caption="A photograph of a red reagent vial with a printed lot number.",
        ),
        "b.png": None,
    }

    updated, n_ok, n_fail = rewrite_markdown(text, descriptions)

    assert "![A red vial.](images/a.png)" in updated
    assert (
        "> [photograph] A photograph of a red reagent vial with a printed lot number."
        in updated
    )
    assert "![](images/b.png)" in updated  # failed description left byte-identical
    assert "![image](image_1.png)" in updated  # unresolved placeholder untouched
    assert n_ok == 1
    assert n_fail == 1


def test_rewrite_markdown_preserves_list_item_indentation():
    text = "- Atemschutz:\n\n  ![](images/a.png)\n\n  Erforderlich bei ...\n"
    descriptions = {
        "a.png": ImageDescription(
            image_type=ImageType.PICTOGRAM,
            alt_text="Respirator pictogram.",
            caption="GHS-style pictogram indicating a respirator (filter P3) is required.",
        ),
    }

    updated, n_ok, n_fail = rewrite_markdown(text, descriptions)

    assert "  ![Respirator pictogram.](images/a.png)" in updated
    assert (
        "  > [pictogram] GHS-style pictogram indicating a respirator (filter P3) is required."
        in updated
    )
    assert n_ok == 1
    assert n_fail == 0


def test_is_local_server():
    assert _is_local_server("hosted_vllm/lightonai/LightOnOCR-2-1B") is True
    # Bare "openai/" is ambiguous (also a hosted LiteLLM provider) and is
    # disambiguated by resolve_litellm_model, not _is_local_server.
    assert _is_local_server("openai/gpt-4o-mini") is False
    assert _is_local_server("anthropic/claude-3-5-sonnet") is False


def test_resolve_litellm_model_prefixes_bare_name_served_locally(monkeypatch):
    monkeypatch.setattr(
        caption_mod, "list_served_models", lambda base_url: ["nvidia/Nemotron-x"]
    )
    assert (
        resolve_litellm_model("nvidia/Nemotron-x", "http://fake")
        == "hosted_vllm/nvidia/Nemotron-x"
    )


def test_resolve_litellm_model_keeps_explicit_prefix_when_served(monkeypatch):
    monkeypatch.setattr(
        caption_mod, "list_served_models", lambda base_url: ["nvidia/Nemotron-x"]
    )
    assert (
        resolve_litellm_model("hosted_vllm/nvidia/Nemotron-x", "http://fake")
        == "hosted_vllm/nvidia/Nemotron-x"
    )


def test_resolve_litellm_model_local_server_wins_over_hosted_provider_name(
    monkeypatch,
):
    # "openai/gpt-oss-20b" is both a valid HF repo and a LiteLLM provider
    # string. When the local server actually serves it, that wins.
    monkeypatch.setattr(
        caption_mod, "list_served_models", lambda base_url: ["openai/gpt-oss-20b"]
    )
    assert (
        resolve_litellm_model("openai/gpt-oss-20b", "http://fake")
        == "hosted_vllm/openai/gpt-oss-20b"
    )


def test_resolve_litellm_model_falls_back_to_hosted_provider(monkeypatch):
    def unreachable(base_url: str) -> list[str]:
        raise RuntimeError("Cannot reach vLLM server at http://fake: boom")

    monkeypatch.setattr(caption_mod, "list_served_models", unreachable)
    assert (
        resolve_litellm_model("anthropic/claude-3-5-sonnet", "http://fake")
        == "anthropic/claude-3-5-sonnet"
    )


def test_resolve_litellm_model_raises_for_unknown_bare_name(monkeypatch):
    monkeypatch.setattr(
        caption_mod, "list_served_models", lambda base_url: ["nvidia/Nemotron-x"]
    )
    with pytest.raises(RuntimeError, match="nvidia/Nemotron-x"):
        resolve_litellm_model("made-up/model", "http://fake")


def test_resolve_litellm_model_raises_for_unavailable_explicit_prefix(monkeypatch):
    monkeypatch.setattr(caption_mod, "list_served_models", lambda base_url: [])
    with pytest.raises(RuntimeError, match="not found on vLLM server"):
        resolve_litellm_model("hosted_vllm/made-up/model", "http://fake")


class FakeArgs:
    model = "hosted_vllm/fake-model"
    base_url = "http://fake"
    api_key = None
    max_tokens = 64
    temperature = 0.2
    max_image_dimension = 32
    max_retries = 1


class FakeCaptionClient:
    """Duck-typed stand-in for instructor.AsyncInstructor: returns a fixed description."""

    async def create(self, **kwargs) -> ImageDescription:
        return ImageDescription(
            image_type=ImageType.PHOTOGRAPH,
            alt_text="a small red square",
            caption="A small solid red square used as a test fixture.",
        )


def test_process_file_captions_images_and_writes_output(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(images_dir / "a.png")

    md_path = tmp_path / "doc.md"
    md_path.write_text("intro\n\n![](images/a.png)\n\noutro\n", encoding="utf-8")
    out_path = tmp_path / "captioned" / "doc.md"
    out_path.parent.mkdir()

    sem = asyncio.Semaphore(2)
    result = asyncio.run(
        process_file(
            FakeCaptionClient(), md_path, images_dir, out_path, sem, FakeArgs()
        )
    )

    assert result == {"images": 1, "captioned": 1, "failed": 0}
    written = out_path.read_text(encoding="utf-8")
    assert "![a small red square](images/a.png)" in written
    assert "> [photograph] A small solid red square used as a test fixture." in written


def test_process_file_with_no_images_copies_text_unchanged(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    md_path = tmp_path / "doc.md"
    md_path.write_text("no images here\n", encoding="utf-8")
    out_path = tmp_path / "captioned" / "doc.md"
    out_path.parent.mkdir()

    sem = asyncio.Semaphore(2)
    result = asyncio.run(
        process_file(
            FakeCaptionClient(), md_path, images_dir, out_path, sem, FakeArgs()
        )
    )

    assert result == {"images": 0, "captioned": 0, "failed": 0}
    assert out_path.read_text(encoding="utf-8") == "no images here\n"
