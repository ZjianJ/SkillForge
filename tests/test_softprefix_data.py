from __future__ import annotations

import json

import numpy as np

from skillopt.softprefix.data import PrefixBatchCollator, TextTrajectoryPrefixDataset


class FakeChatTokenizer:
    chat_template = "fake-template"
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.tools_seen = []

    def apply_chat_template(self, messages, **kwargs):
        self.tools_seen.append(kwargs.get("tools"))
        rendered = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            rendered.append(f"<{role}>{content}")
            tool_calls = message.get("tool_calls") or []
            for tool_call in tool_calls:
                function = tool_call["function"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                rendered.append(f"<call>{function['name']}:{arguments}")
        if kwargs.get("add_generation_prompt"):
            rendered.append("<assistant>")
        return "".join(rendered)

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [ord(char) for char in text]}


def test_text_trajectory_dataset_masks_prompt_and_supervises_target_message() -> None:
    tokenizer = FakeChatTokenizer()
    item = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        "target_message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "officeqa_tool_1_1",
                    "type": "function",
                    "function": {
                        "name": "grep",
                        "arguments": json.dumps({"pattern": "debt", "path": "/tmp/doc.txt"}),
                    },
                }
            ],
        },
    }
    dataset = TextTrajectoryPrefixDataset(
        [item],
        tokenizer,
        max_prompt_tokens=4096,
        max_target_tokens=4096,
    )

    encoded = dataset[0]
    prompt = tokenizer.apply_chat_template(
        item["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        tools=tokenizer.tools_seen[0],
    )
    full = tokenizer.apply_chat_template(
        item["messages"] + [item["target_message"]],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
        tools=tokenizer.tools_seen[0],
    ) + tokenizer.eos_token
    target_ids = [ord(char) for char in full[len(prompt):]]

    assert encoded.input_ids == [ord(char) for char in full]
    assert encoded.labels == [-100] * len(prompt) + target_ids
    assert tokenizer.tools_seen[0] is not None


def test_text_trajectory_dataset_sets_skill_section_insert_idx() -> None:
    tokenizer = FakeChatTokenizer()
    item = {
        "messages": [
            {"role": "system", "content": "system\n## Skill\n<|skillopt_soft_prefix_insert|>\n\nrules"},
            {"role": "user", "content": "question"},
        ],
        "target": "answer",
    }
    dataset = TextTrajectoryPrefixDataset(
        [item],
        tokenizer,
        max_prompt_tokens=4096,
        max_target_tokens=4096,
        injection_position="skill_section",
    )

    encoded = dataset[0]
    rendered_prompt = tokenizer.apply_chat_template(
        item["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    marker_idx = rendered_prompt.index("<|skillopt_soft_prefix_insert|>")
    clean_prompt = rendered_prompt.replace("<|skillopt_soft_prefix_insert|>", "", 1)

    assert encoded.prefix_insert_idx == marker_idx
    assert encoded.input_ids[: len(clean_prompt)] == [ord(char) for char in clean_prompt]
    assert "<|skillopt_soft_prefix_insert|>" not in "".join(chr(token_id) for token_id in encoded.input_ids)


def test_text_trajectory_dataset_masks_selected_target_positions_and_keeps_eos() -> None:
    tokenizer = FakeChatTokenizer()
    item = {
        "id": "selective-example",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        "target": "abcd",
        "selected_indices": [1, 3],
    }
    dataset = TextTrajectoryPrefixDataset(
        [item],
        tokenizer,
        max_prompt_tokens=4096,
        max_target_tokens=4096,
        selective_label_field="selected_indices",
        always_supervise_eos=True,
    )

    encoded = dataset[0]
    target_ids = [ord(char) for char in "abcd<eos>"]
    target_labels = encoded.labels[-len(target_ids):]
    assert target_labels[0] == -100
    assert target_labels[1] == ord("b")
    assert target_labels[3] == ord("d")
    assert target_labels[-1] == ord(">")
    assert sum(label != -100 for label in target_labels) == 3


def test_text_trajectory_dataset_loads_stage_start_retention_topk(tmp_path) -> None:
    tokenizer = FakeChatTokenizer()
    target_ids = [ord(char) for char in "abcd<eos>"]
    topk_ids = np.tile(np.array([[1, 2]], dtype=np.int32), (len(target_ids), 1))
    topk_logp = np.tile(
        np.log(np.array([[0.5, 0.25]], dtype=np.float32)),
        (len(target_ids), 1),
    )
    residual = np.full(len(target_ids), np.log(0.25), dtype=np.float32)
    cache = tmp_path / "stage_start.npz"
    np.savez_compressed(
        cache,
        target_ids=np.asarray(target_ids, dtype=np.int32),
        current_topk_ids=topk_ids,
        current_topk_logp=topk_logp,
        current_residual_log_mass=residual,
    )
    item = {
        "id": "retention-example",
        "messages": [{"role": "user", "content": "question"}],
        "target": "abcd",
        "replay_indices": [1, 3],
        "retention_cache": str(cache),
    }
    dataset = TextTrajectoryPrefixDataset(
        [item],
        tokenizer,
        max_prompt_tokens=4096,
        max_target_tokens=4096,
        retention_loss_weight=1.0,
    )
    encoded = dataset[0]
    prompt_length = len(encoded.input_ids) - len(target_ids)
    assert encoded.extra_inputs is not None
    assert encoded.extra_inputs["retention_logit_indices"] == [
        prompt_length,
        prompt_length + 2,
    ]
    batch = PrefixBatchCollator(tokenizer.pad_token_id)([encoded])
    assert batch["retention_mask"].tolist() == [[True, True]]
    assert batch["retention_topk_ids"].tolist() == [[[1, 2], [1, 2]]]
