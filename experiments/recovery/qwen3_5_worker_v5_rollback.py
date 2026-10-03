#!/usr/bin/env python3
"""Qwen3.5 worker for visual task-stage regression judgments.

The worker receives RGB evidence and semantic stage descriptions only.  The
simulator oracle target is deliberately absent from the request payload.
"""

from __future__ import annotations

import re
from typing import Any

from PIL import Image

try:
    from stage_reward import qwen3_5_worker_v3 as base
except ImportError:  # Direct script execution places this directory on sys.path.
    import qwen3_5_worker_v3 as base


def parse_stage_prediction(text: str) -> str:
    value = text.strip().lower()
    stage = re.search(r"\bstage[_\s-]?(\d+)\b", value)
    if stage is not None:
        return f"stage_{int(stage.group(1))}"
    if re.search(r"\bunchanged\b", value):
        return "unchanged"
    return "unknown"


def gate_messages(
    items: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[Image.Image]]:
    messages: list[list[dict[str, Any]]] = []
    images: list[Image.Image] = []
    system = (
        "You are a visual task-stage regression judge for robot manipulation. "
        "The first RGB image is an anchor captured when the stated current "
        "stage was valid. The second RGB image is a later observation. A "
        "disturbance may have invalidated one or more persistent prerequisites. "
        "Choose the earliest listed stage that the robot must execute next to "
        "finish the task. If the current stage remains valid, answer unchanged. "
        "Use only visible RGB evidence and the stage descriptions. Do not infer "
        "hidden contacts. Return exactly one label: stage_0, stage_1, ..., "
        "unchanged, or unknown. Return unknown only when visibility is genuinely "
        "insufficient. Do not explain your answer."
    )
    for item in items:
        anchor = base.decode_jpeg(str(item["reference_jpeg_b64"]))
        current = base.decode_jpeg(str(item["current_jpeg_b64"]))
        images.extend((anchor, current))
        stage_lines = "\n".join(
            f"stage_{index}: {description}"
            for index, description in enumerate(item["stage_descriptions"])
        )
        instruction = (
            f"Task: {item['task_instruction']}\n"
            f"Stage before the possible regression: stage_{item['stage_id']}\n"
            "Ordered stages (stage_i means this is the next behavior the robot "
            f"should execute):\n{stage_lines}\n"
            "Compare the later image with the valid anchor and output the next "
            "stage label."
        )
        messages.append(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": anchor},
                        {"type": "text", "text": "Valid-stage anchor image."},
                        {"type": "image", "image": current},
                        {"type": "text", "text": instruction},
                    ],
                },
            ]
        )
    return messages, images


base.gate_messages = gate_messages
base.parse_gate_prediction = parse_stage_prediction


if __name__ == "__main__":
    base.main()
