"""Tests for DocVQA soft-prefix image budgeting."""
from __future__ import annotations

from skillopt.softprefix.data import apply_docvqa_image_budget, resolve_docvqa_image_token_budget


def test_auto_docvqa_image_budget_leaves_room_for_text() -> None:
    assert resolve_docvqa_image_token_budget(max_prompt_tokens=16_384, configured_max_image_tokens=0) == 12_288


def test_configured_docvqa_image_budget_is_clamped_to_prompt_budget() -> None:
    assert resolve_docvqa_image_token_budget(max_prompt_tokens=8_192, configured_max_image_tokens=20_000) == 8_064


def test_apply_docvqa_image_budget_sets_qwen_pixel_cap() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "file:///tmp/page.png"},
                {"type": "text", "text": "question"},
            ],
        },
    ]

    updated = apply_docvqa_image_budget(messages, max_image_tokens=100, image_patch_size=14)

    image = updated[1]["content"][0]
    assert image["max_pixels"] == 100 * (14 * 2) ** 2
    assert image["image"] == "file:///tmp/page.png"
    assert "max_pixels" not in messages[1]["content"][0]
