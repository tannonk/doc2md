"""Smoke test: a live vLLM server accepts and understands image input.

Sends a solid red square to the server's OpenAI-compatible chat endpoint and
checks that the reply names the colour. Skipped automatically when no server
is reachable, so it is safe in offline runs.

Environment variables:
    VLLM_BASE_URL  Server URL (default: DEFAULT_BASE_URL of the captioning stage).
    VLLM_MODEL     Model name to query (default: first model the server lists).

Usage:
    VLLM_BASE_URL=http://localhost:8000 pytest tests/test_vllm_vision_smoke.py -v
    pytest -m "not vllm"   # leave live-server tests out
"""

import os

import pytest
import requests
from PIL import Image

from doc2md.caption import DEFAULT_BASE_URL
from doc2md.resolve import image_to_data_uri

pytestmark = pytest.mark.vllm

PROMPT = "What colour is this image? Answer with one word."
REQUEST_TIMEOUT_S = 120


@pytest.fixture(scope="module")
def api_base() -> str:
    """OpenAI-style API root (ending in /v1) of the server under test."""
    base_url = os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


@pytest.fixture(scope="module")
def model_name(api_base: str) -> str:
    """Model to query; skips the whole module if no server answers."""
    try:
        response = requests.get(f"{api_base}/models", timeout=5)
        response.raise_for_status()
    except requests.RequestException as error:
        pytest.skip(f"no vLLM server at {api_base}: {error}")
    return os.environ.get("VLLM_MODEL") or response.json()["data"][0]["id"]


@pytest.fixture(scope="module")
def reply(api_base: str, model_name: str) -> requests.Response:
    """Server response to 'what colour is this?' about a pure red image."""
    red_square = Image.new("RGB", (512, 512), color=(255, 0, 0))
    payload = {
        "model": model_name,
        "temperature": 0,
        # Generous: reasoning models may think before answering.
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_uri(red_square)},
                    },
                ],
            }
        ],
    }
    response = requests.post(
        f"{api_base}/chat/completions", json=payload, timeout=REQUEST_TIMEOUT_S
    )
    if response.status_code == 400 and "out of memory" in response.text:
        pytest.fail(
            "Server ran out of GPU/host memory while preprocessing the image. "
            "Free memory or lower GPU_MEMORY_UTILIZATION; see docs/gpu_memory_budget.md."
        )
    return response


def test_server_accepts_image_input(reply: requests.Response) -> None:
    assert reply.status_code == 200, reply.text
    assert reply.json()["choices"][0]["message"]["content"]


def test_model_sees_the_image(reply: requests.Response) -> None:
    answer = reply.json()["choices"][0]["message"]["content"]
    assert "red" in answer.lower(), f"model did not see a red image: {answer!r}"
