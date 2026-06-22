"""Tests for vLLM-backed soft-prefix inference glue."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
import urllib.request


def _read_http_json(url: str, *, payload: dict | None = None) -> dict:
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=3) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_vllm_prompt_embeds_module():
    module_path = Path(__file__).parents[1] / "skillopt" / "softprefix" / "vllm_prompt_embeds.py"
    spec = importlib.util.spec_from_file_location("test_vllm_prompt_embeds_direct", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_softprefix_settings_parse_vllm_inference_backend() -> None:
    from skillopt.softprefix.trainer import SoftPrefixSettings

    settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "inference_backend": "vllm_prompt_embeds",
            "inference_base_url": "http://127.0.0.1:8010",
            "inference_timeout_seconds": 12.5,
        }
    )

    assert settings.inference_backend == "vllm_prompt_embeds"
    assert settings.inference_base_url == "http://127.0.0.1:8010"
    assert settings.inference_timeout_seconds == 12.5


def test_officeqa_prompt_prefers_resolved_source_paths(monkeypatch) -> None:
    from skillopt.softprefix import data

    captured: dict[str, object] = {}

    class FakeTokenizer:
        chat_template = None

    def fake_build_user(item, candidate_files=None, **kwargs):
        captured["candidate_files"] = candidate_files
        return "user prompt"

    monkeypatch.setattr(data, "_build_officeqa_system", lambda *args, **kwargs: "system prompt")
    monkeypatch.setattr(data, "_build_officeqa_user", fake_build_user)

    data.build_officeqa_prompt(
        FakeTokenizer(),
        {
            "question": "q",
            "source_files": ["treasury_bulletin_1941_01.txt"],
            "resolved_source_paths": ["/abs/docs/treasury_bulletin_1941_01.txt"],
        },
    )

    assert captured["candidate_files"] == ["/abs/docs/treasury_bulletin_1941_01.txt"]


def test_officeqa_dataset_resolves_training_source_paths(monkeypatch, tmp_path) -> None:
    from skillopt.softprefix import trainer
    from skillopt.softprefix.trainer import SoftPrefixSettings

    class FakePrefixModel:
        tokenizer = object()

    monkeypatch.setattr(trainer, "resolve_docs_roots", lambda data_dirs=None: [str(tmp_path)])
    monkeypatch.setattr(
        trainer,
        "resolve_candidate_files",
        lambda source_files, docs_roots: [str(tmp_path / source_files[0])],
    )

    dataset = trainer._build_dataset(
        "officeqa",
        [{"id": "uid0", "question": "q", "source_files": ["doc.txt"], "ground_truth": "gold"}],
        FakePrefixModel(),
        {"data_dirs": [str(tmp_path)]},
        SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B", "training_data": "supervised"}),
    )

    assert dataset.items[0]["resolved_source_paths"] == [str(tmp_path / "doc.txt")]


def test_officeqa_glob_accepts_absolute_allowed_candidate_path(tmp_path) -> None:
    from skillopt.envs.officeqa.tool_runtime import run_tool

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    doc_path = docs_root / "treasury_bulletin_1954_02.txt"
    doc_path.write_text("evidence", encoding="utf-8")

    _cmd, obs = run_tool(
        "glob",
        {"pattern": str(doc_path)},
        allowed_roots=[str(docs_root)],
        allowed_files=[doc_path.name],
    )

    assert obs == str(doc_path)


def test_officeqa_grep_missing_malformed_path_returns_tool_error(tmp_path) -> None:
    from skillopt.envs.officeqa.tool_runtime import run_tool

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    doc_path = docs_root / "treasury_bulletin_2007_09.txt"
    doc_path.write_text("public debt evidence", encoding="utf-8")
    malformed_path = (
        str(docs_root / "treasury_bullet")
        + "\n\n<tool_call>\n<function=grep>\n<parameter=path>\n"
        + str(doc_path)
    )

    _cmd, obs = run_tool(
        "grep",
        {"pattern": "debt", "path": malformed_path},
        allowed_roots=[str(docs_root)],
        allowed_files=[],
    )

    assert obs == "[grep error: path not found]"


def test_officeqa_read_missing_allowed_path_returns_tool_error(tmp_path) -> None:
    from skillopt.envs.officeqa.tool_runtime import run_tool

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    missing_path = docs_root / "missing.txt"

    _cmd, obs = run_tool(
        "read",
        {"path": str(missing_path)},
        allowed_roots=[str(docs_root)],
        allowed_files=[],
    )

    assert obs == "[read error: path not found]"


def test_vllm_client_updates_prefix_and_generates_prompts(monkeypatch) -> None:
    from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmClient

    requests: list[tuple[str, dict]] = []

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        requests.append((req.full_url, payload))
        if req.full_url.endswith("/generate"):
            return FakeResponse({"texts": ["<action>look</action>"]})
        return FakeResponse({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = SoftPrefixVllmClient("http://localhost:8010", timeout_seconds=3)
    client.set_prefix([[1.0, 2.0], [3.0, 4.0]])
    texts = client.generate_from_prompts(
        ["prompt"],
        max_prompt_tokens=128,
        max_new_tokens=8,
        temperature=0.0,
    )

    assert texts == ["<action>look</action>"]
    assert requests == [
        (
            "http://localhost:8010/set_prefix",
            {"prefix_embeddings": [[1.0, 2.0], [3.0, 4.0]]},
        ),
        (
            "http://localhost:8010/generate",
            {
                "prompts": ["prompt"],
                "max_prompt_tokens": 128,
                "max_new_tokens": 8,
                "temperature": 0.0,
            },
        ),
    ]


def test_vllm_client_sends_prefix_insert_indices(monkeypatch) -> None:
    from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmClient

    requests: list[tuple[str, dict]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"texts": ["answer"]}).encode("utf-8")

    def fake_urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        requests.append((req.full_url, payload))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = SoftPrefixVllmClient("http://localhost:8010", timeout_seconds=3)
    texts = client.generate_from_prompts(
        ["prompt"],
        max_prompt_tokens=128,
        max_new_tokens=8,
        temperature=0.0,
        prefix_insert_indices=[3],
    )

    assert texts == ["answer"]
    assert requests == [
        (
            "http://localhost:8010/generate",
            {
                "prompts": ["prompt"],
                "max_prompt_tokens": 128,
                "max_new_tokens": 8,
                "temperature": 0.0,
                "prefix_insert_indices": [3],
            },
        )
    ]


def test_vllm_client_sets_prefix_injection_position(monkeypatch) -> None:
    from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmClient

    requests: list[tuple[str, dict]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"ok": True}).encode("utf-8")

    def fake_urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        requests.append((req.full_url, payload))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = SoftPrefixVllmClient("http://localhost:8010", timeout_seconds=3)
    client.set_prefix([[1.0, 2.0]], injection_position="skill_section")

    assert requests == [
        (
            "http://localhost:8010/set_prefix",
            {
                "prefix_embeddings": [[1.0, 2.0]],
                "prefix_injection_position": "skill_section",
            },
        )
    ]


def test_local_prompt_generation_reports_progress(monkeypatch) -> None:
    from skillopt.softprefix import trainer

    progress_calls: list[tuple[list[str], str, str, bool]] = []

    def fake_tqdm(iterable, *, desc, unit, leave):
        items = list(iterable)
        progress_calls.append((items, desc, unit, leave))
        return items

    class FakePrefixModel:
        def generate_from_prompt(self, prompt: str, **kwargs) -> str:
            return f"response:{prompt}"

    monkeypatch.setattr(trainer, "tqdm", fake_tqdm)

    responses = trainer._generate_prompt_responses(
        FakePrefixModel(),
        None,
        ["prompt-a", "prompt-b"],
        max_prompt_tokens=128,
        max_new_tokens=16,
        temperature=0.0,
        desc="  Init Test",
    )

    assert responses == ["response:prompt-a", "response:prompt-b"]
    assert progress_calls == [(["prompt-a", "prompt-b"], "  Init Test Generate", "ex", False)]


def test_vllm_client_requests_prefix_chat_completion(monkeypatch) -> None:
    from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmClient

    requests: list[tuple[str, dict]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "read", "arguments": "{\"path\": \"doc.txt\"}"},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        requests.append((req.full_url, payload))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = SoftPrefixVllmClient("http://localhost:8010", timeout_seconds=3)
    message, metadata = client.generate_chat_completion(
        [{"role": "user", "content": "find it"}],
        max_prompt_tokens=128,
        max_new_tokens=16,
        temperature=0.0,
        tools=[{"type": "function", "function": {"name": "read"}}],
        tool_choice="auto",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert message["tool_calls"][0]["function"]["name"] == "read"
    assert metadata == {"finish_reason": "tool_calls"}
    assert requests == [
        (
            "http://localhost:8010/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "find it"}],
                "max_prompt_tokens": 128,
                "max_tokens": 16,
                "temperature": 0.0,
                "tools": [{"type": "function", "function": {"name": "read"}}],
                "tool_choice": "auto",
                "chat_template_kwargs": {"enable_thinking": False},
                "use_prefix": True,
            },
        )
    ]


def test_vllm_engine_accepts_empty_prefix_for_plain_baseline() -> None:
    import torch

    from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmEngine

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, prompt, **kwargs):
            del prompt, kwargs
            return {"input_ids": torch.tensor([[10, 11]])}

    class FakeEmbedding:
        weight = torch.zeros((20, 2), dtype=torch.float32)

        def __call__(self, input_ids):
            del input_ids
            return torch.tensor([[[10.0, 10.5], [11.0, 11.5]]])

    engine = SoftPrefixVllmEngine.__new__(SoftPrefixVllmEngine)
    engine.torch = torch
    engine.tokenizer = FakeTokenizer()
    engine.embedding_layer = FakeEmbedding()
    engine.embedding_device = "cpu"
    engine.set_prefix([])

    assert engine.prefix_embeddings.shape == (0, 2)
    prompt_input = engine._prompt_input("prompt", max_prompt_tokens=128)
    assert prompt_input["prompt_embeds"].tolist() == [[10.0, 10.5], [11.0, 11.5]]
    assert prompt_input["prompt_token_ids"] == [10, 11]


def test_vllm_engine_forwards_tensor_parallel_size(monkeypatch) -> None:
    import sys
    import types

    import torch

    from skillopt.softprefix import vllm_prompt_embeds

    captured: dict[str, object] = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            del model_name, kwargs
            return FakeTokenizer()

    class FakeEmbedding:
        weight = torch.zeros((20, 2), dtype=torch.float32)

    class FakeLlm:
        def __init__(self, **kwargs) -> None:
            captured["llm_kwargs"] = kwargs
            self.model_config = object()

    fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    fake_vllm = types.SimpleNamespace(LLM=FakeLlm, SamplingParams=lambda **kwargs: kwargs)
    fake_registry = types.SimpleNamespace(supports_multimodal_inputs=lambda model_config: False)
    fake_multimodal_registry = types.SimpleNamespace(MULTIMODAL_REGISTRY=fake_registry)

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.multimodal.registry", fake_multimodal_registry)
    monkeypatch.setattr(vllm_prompt_embeds, "_load_embedding_layer", lambda **kwargs: FakeEmbedding())

    vllm_prompt_embeds.SoftPrefixVllmEngine(
        model_name="fake/model",
        tensor_parallel_size=2,
    )

    assert captured["llm_kwargs"]["tensor_parallel_size"] == 2


def test_vllm_engine_prompt_input_includes_token_ids_for_mrope() -> None:
    import torch

    from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmEngine

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, prompt, **kwargs):
            del prompt, kwargs
            return {"input_ids": torch.tensor([[10, 11]])}

    class FakeEmbedding:
        weight = torch.zeros((20, 2), dtype=torch.float32)

        def __call__(self, input_ids):
            del input_ids
            return torch.tensor([[[10.0, 10.5], [11.0, 11.5]]])

    engine = SoftPrefixVllmEngine.__new__(SoftPrefixVllmEngine)
    engine.torch = torch
    engine.tokenizer = FakeTokenizer()
    engine.embedding_layer = FakeEmbedding()
    engine.embedding_device = "cpu"
    engine.prefix_embeddings = torch.tensor([[1.0, 1.5], [2.0, 2.5]])

    prompt_input = engine._prompt_input("prompt", max_prompt_tokens=128)

    assert prompt_input["prompt_embeds"].tolist() == [
        [1.0, 1.5],
        [2.0, 2.5],
        [10.0, 10.5],
        [11.0, 11.5],
    ]
    assert prompt_input["prompt_token_ids"] == [0, 0, 10, 11]
    assert prompt_input["prompt_is_token_ids"] == [False, False, True, True]


def test_vllm_engine_prompt_input_can_insert_prefix_in_middle() -> None:
    import torch

    from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmEngine

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, prompt, **kwargs):
            del prompt, kwargs
            return {"input_ids": torch.tensor([[10, 11, 12]])}

    class FakeEmbedding:
        weight = torch.zeros((20, 2), dtype=torch.float32)

        def __call__(self, input_ids):
            del input_ids
            return torch.tensor([[[10.0, 10.5], [11.0, 11.5], [12.0, 12.5]]])

    engine = SoftPrefixVllmEngine.__new__(SoftPrefixVllmEngine)
    engine.torch = torch
    engine.tokenizer = FakeTokenizer()
    engine.embedding_layer = FakeEmbedding()
    engine.embedding_device = "cpu"
    engine.supports_mm_inputs = False
    engine.prefix_embeddings = torch.tensor([[1.0, 1.5], [2.0, 2.5]])

    prompt_input = engine._prompt_input("prompt", max_prompt_tokens=128, prefix_insert_idx=1)

    assert prompt_input["prompt_embeds"].tolist() == [
        [10.0, 10.5],
        [1.0, 1.5],
        [2.0, 2.5],
        [11.0, 11.5],
        [12.0, 12.5],
    ]
    assert prompt_input["prompt_token_ids"] == [10, 0, 0, 11, 12]
    assert prompt_input["prompt_is_token_ids"] == [True, False, False, True, True]


def test_vllm_engine_chat_messages_insert_skill_section_marker() -> None:
    import torch

    from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmEngine

    class FakeTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return (
                f"SYS:{messages[0]['content']}\n"
                f"USER:{messages[1]['content']}\n"
                "ASSISTANT:"
            )

        def __call__(self, text, **kwargs):
            del kwargs
            if isinstance(text, list):
                return {"input_ids": torch.tensor([[ord(ch) for ch in text[0]]])}
            return {"input_ids": torch.tensor([[ord(ch) for ch in text]])}

    class FakeEmbedding:
        weight = torch.zeros((256, 2), dtype=torch.float32)

        def __call__(self, input_ids):
            values = input_ids[0].to(dtype=torch.float32)
            return torch.stack([values, values + 0.5], dim=-1).unsqueeze(0)

    class FakeOutput:
        outputs = [type("FakeCompletion", (), {"text": "ok"})()]

    class FakeLlm:
        def __init__(self) -> None:
            self.prompt_inputs = []

        def generate(self, prompt_inputs, sampling):
            del sampling
            self.prompt_inputs = prompt_inputs
            return [FakeOutput()]

    engine = SoftPrefixVllmEngine.__new__(SoftPrefixVllmEngine)
    engine.torch = torch
    engine.SamplingParams = lambda **kwargs: kwargs
    engine.tokenizer = FakeTokenizer()
    engine.embedding_layer = FakeEmbedding()
    engine.embedding_device = "cpu"
    engine.language_model_only = True
    engine.llm = FakeLlm()
    engine.set_prefix([[1.0, 1.5]], injection_position="skill_section")

    text = engine.generate_from_messages(
        [
            {"role": "system", "content": "intro\n{skill_section}rules"},
            {"role": "user", "content": "question"},
        ],
        max_prompt_tokens=256,
        max_new_tokens=8,
    )

    prompt_input = engine.llm.prompt_inputs[0]
    rendered_prompt = "SYS:intro\nrules\nUSER:question\nASSISTANT:"
    insert_idx = len("SYS:intro\n")
    expected_token_ids = (
        [ord(ch) for ch in rendered_prompt[:insert_idx]]
        + [0]
        + [ord(ch) for ch in rendered_prompt[insert_idx:]]
    )

    assert text == "ok"
    assert prompt_input["prompt_token_ids"] == expected_token_ids
    assert prompt_input["prompt_embeds"][insert_idx].tolist() == [1.0, 1.5]
    assert "{skill_section}" not in "".join(chr(token_id) for token_id in prompt_input["prompt_token_ids"] if token_id)


def test_vllm_engine_renders_prior_tool_calls_with_mapping_arguments() -> None:
    from skillopt.softprefix.vllm_prompt_embeds import SoftPrefixVllmEngine

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            tool_call = messages[1]["tool_calls"][0]
            assert tool_call["function"]["arguments"] == {"path": "doc.txt"}
            return "rendered"

    engine = SoftPrefixVllmEngine.__new__(SoftPrefixVllmEngine)
    engine.tokenizer = FakeTokenizer()

    prompt = engine._messages_to_prompt(
        [
            {"role": "user", "content": "find it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{\"path\": \"doc.txt\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "needle evidence"},
        ],
        tools=[{"type": "function", "function": {"name": "read"}}],
        chat_template_kwargs={"enable_thinking": False},
    )

    assert prompt == "rendered"


def test_alfworld_eval_batches_active_prompts_with_external_generator(tmp_path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    class FakeTokenizer:
        pass

    class FakePrefixModel:
        tokenizer = FakeTokenizer()

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def generate_from_prompts(self, prompts, *, max_prompt_tokens, max_new_tokens, temperature):
            self.calls.append(list(prompts))
            return [
                f"<think>ok {idx}</think><action>look</action>"
                for idx, _prompt in enumerate(prompts)
            ]

    class FakeEnv:
        def __init__(self) -> None:
            self.step_calls = 0
            self.closed = False

        def reset(self, _):
            return (
                {
                    "text": ["obs 0", "obs 1"],
                    "anchor": [
                        "Your task is to: task 0",
                        "Your task is to: task 1",
                    ],
                },
                [
                    {"extra.gamefile": "/dummy/valid_seen/0/game.tw-pddl"},
                    {"extra.gamefile": "/dummy/valid_seen/1/game.tw-pddl"},
                ],
            )

        def step(self, actions):
            self.step_calls += 1
            assert actions == [
                "<think>ok 0</think><action>look</action>",
                "<think>ok 1</think><action>look</action>",
            ]
            return (
                {"anchor": ["done 0", "done 1"]},
                [0.0, 1.0],
                [True, True],
                [{"won": False}, {"won": True}],
            )

        def close(self):
            self.closed = True

    def fake_template(tokenizer, messages, enable_thinking=False):
        del tokenizer, enable_thinking
        return "\n".join(message["content"] for message in messages)

    fake_env = FakeEnv()
    fake_generator = FakeGenerator()
    monkeypatch.setattr(trainer, "build_alfworld_env", lambda **kwargs: fake_env)
    monkeypatch.setattr(trainer, "_apply_text_chat_template", fake_template)

    hard, soft, results = trainer.evaluate_alfworld_prefix(
        FakePrefixModel(),
        [
            {"id": "env0", "gamefile": "/dummy/valid_seen/0/game.tw-pddl"},
            {"id": "env1", "gamefile": "/dummy/valid_seen/1/game.tw-pddl"},
        ],
        out_dir=str(tmp_path),
        max_steps=3,
        max_prompt_tokens=128,
        max_new_tokens=8,
        temperature=0.0,
        generator=fake_generator,
    )

    assert fake_generator.calls == [
        [
            "You are an expert agent operating in the ALFRED Embodied Environment.\nobs 0",
            "You are an expert agent operating in the ALFRED Embodied Environment.\nobs 1",
        ]
    ]
    assert hard == 0.5
    assert soft == 0.5
    assert [row["hard"] for row in results] == [0, 1]
    assert fake_env.closed is True


def test_alfworld_eval_keeps_invalid_model_response_without_fallback(tmp_path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    class FakeTokenizer:
        pass

    class FakePrefixModel:
        tokenizer = FakeTokenizer()

    class FakeGenerator:
        def generate_from_prompts(self, prompts, *, max_prompt_tokens, max_new_tokens, temperature):
            del prompts, max_prompt_tokens, max_new_tokens, temperature
            return ["plain text without action tag"]

    class FakeEnv:
        def reset(self, _):
            return (
                {
                    "text": ["obs 0"],
                    "anchor": ["Your task is to: task 0"],
                },
                [{"extra.gamefile": "/dummy/valid_seen/0/game.tw-pddl"}],
            )

        def step(self, actions):
            assert actions == ["plain text without action tag"]
            return (
                {"anchor": ["invalid action"]},
                [0.0],
                [True],
                [{"won": False}],
            )

    def fake_template(tokenizer, messages, enable_thinking=False):
        del tokenizer, enable_thinking
        return "\n".join(message["content"] for message in messages)

    monkeypatch.setattr(trainer, "build_alfworld_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(trainer, "_apply_text_chat_template", fake_template)

    trainer.evaluate_alfworld_prefix(
        FakePrefixModel(),
        [{"id": "env0", "gamefile": "/dummy/valid_seen/0/game.tw-pddl"}],
        out_dir=str(tmp_path),
        max_steps=1,
        max_prompt_tokens=128,
        max_new_tokens=8,
        temperature=0.0,
        generator=FakeGenerator(),
    )

    conversation_path = tmp_path / "predictions" / "env0" / "conversation.json"
    conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
    assert conversation[0]["model_response"] == "plain text without action tag"


def test_plain_baseline_alfworld_uses_vllm_generator_with_empty_prefix(monkeypatch) -> None:
    import torch

    from skillopt.softprefix import trainer
    from skillopt.softprefix.trainer import SoftPrefixSettings

    class FakePrefixModel:
        prefix_embeddings = torch.zeros(4, 8)

    captured: dict[str, object] = {}

    class FakeVllmClient:
        def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
            captured["base_url"] = base_url
            captured["timeout_seconds"] = timeout_seconds

        def set_prefix(self, prefix_embeddings, *, injection_position=None) -> None:
            captured["prefix_rows"] = len(prefix_embeddings)
            captured["injection_position"] = injection_position

        def generate_from_prompts(self, *args, **kwargs):
            raise AssertionError("evaluate_alfworld_prefix should call the generator")

    def fake_evaluate(prefix_model, items, **kwargs):
        captured["generator"] = kwargs.get("generator")
        captured["prefix_model"] = prefix_model
        return 0.0, 0.0, []

    monkeypatch.setattr("skillopt.softprefix.vllm_prompt_embeds.SoftPrefixVllmClient", FakeVllmClient)
    monkeypatch.setattr(trainer, "evaluate_alfworld_prefix", fake_evaluate)

    settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "inference_backend": "vllm_prompt_embeds",
            "inference_base_url": "http://127.0.0.1:8010",
            "inference_timeout_seconds": 12.5,
        }
    )

    trainer._evaluate_plain_baseline(
        "alfworld",
        FakePrefixModel(),
        [{"id": "env0"}],
        cfg={"max_steps": 3},
        settings=settings,
        out_dir="/tmp/plain",
        desc="  Plain Val",
    )

    assert captured["base_url"] == "http://127.0.0.1:8010"
    assert captured["timeout_seconds"] == 12.5
    assert captured["prefix_rows"] == 0
    assert isinstance(captured["generator"], FakeVllmClient)
    assert hasattr(captured["prefix_model"], "generate_from_prompt")


def test_plain_baseline_officeqa_uses_vllm_generator_with_empty_prefix(monkeypatch) -> None:
    import torch

    from skillopt.softprefix import trainer
    from skillopt.softprefix.trainer import SoftPrefixSettings

    class FakeTokenizer:
        pass

    class FakePrefixModel:
        tokenizer = FakeTokenizer()
        prefix_embeddings = torch.zeros(4, 8)

        def generate_from_prompt(self, *args, **kwargs):
            raise AssertionError("plain OfficeQA eval should use the vLLM generator")

    captured: dict[str, object] = {}

    class FakeVllmClient:
        def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
            captured["base_url"] = base_url
            captured["timeout_seconds"] = timeout_seconds

        def set_prefix(self, prefix_embeddings, *, injection_position=None) -> None:
            captured["prefix_rows"] = len(prefix_embeddings)
            captured["injection_position"] = injection_position

        def generate_from_prompts(self, prompts, *, max_prompt_tokens, max_new_tokens, temperature):
            captured["prompts"] = list(prompts)
            captured["limits"] = (max_prompt_tokens, max_new_tokens, temperature)
            return ["<answer>gold</answer>"]

    monkeypatch.setattr("skillopt.softprefix.vllm_prompt_embeds.SoftPrefixVllmClient", FakeVllmClient)
    monkeypatch.setattr(trainer, "build_officeqa_prompt", lambda tokenizer, item, enable_thinking=False: "office prompt")

    settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "inference_backend": "vllm_prompt_embeds",
            "inference_base_url": "http://127.0.0.1:8010",
            "inference_timeout_seconds": 12.5,
            "max_prompt_tokens": 123,
            "max_new_tokens": 7,
        }
    )

    hard, soft, _ = trainer._evaluate_plain_baseline(
        "officeqa",
        FakePrefixModel(),
        [{"id": "uid0", "ground_truth": "gold"}],
        cfg={},
        settings=settings,
        out_dir="/tmp/plain-officeqa",
        desc="  Plain Val",
    )

    assert (hard, soft) == (1.0, 1.0)
    assert captured["base_url"] == "http://127.0.0.1:8010"
    assert captured["timeout_seconds"] == 12.5
    assert captured["prefix_rows"] == 0
    assert captured["prompts"] == ["office prompt"]
    assert captured["limits"] == (123, 7, 0.0)


def test_livemath_skill_section_uses_vllm_generator_with_insert_indices(monkeypatch, tmp_path) -> None:
    import torch

    from skillopt.softprefix import trainer
    from skillopt.softprefix.trainer import SoftPrefixSettings

    class FakeTokenizer:
        pass

    class FakePrefixModel:
        tokenizer = FakeTokenizer()
        prefix_embeddings = torch.zeros(4, 8)

        def generate_from_prompt(self, *args, **kwargs):
            raise AssertionError("skill-section LiveMath eval should use the vLLM generator")

    captured: dict[str, object] = {}

    class FakeVllmClient:
        def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
            captured["base_url"] = base_url
            captured["timeout_seconds"] = timeout_seconds

        def set_prefix(self, prefix_embeddings, *, injection_position=None) -> None:
            captured["prefix_rows"] = len(prefix_embeddings)
            captured["injection_position"] = injection_position

        def generate_from_prompts(
            self,
            prompts,
            *,
            max_prompt_tokens,
            max_new_tokens,
            temperature,
            prefix_insert_indices=None,
        ):
            captured["prompts"] = list(prompts)
            captured["prefix_insert_indices"] = list(prefix_insert_indices or [])
            captured["limits"] = (max_prompt_tokens, max_new_tokens, temperature)
            return ["<answer>B</answer>"]

    monkeypatch.setattr("skillopt.softprefix.vllm_prompt_embeds.SoftPrefixVllmClient", FakeVllmClient)
    monkeypatch.setattr(
        trainer,
        "build_livemath_prompt_and_insert_idx",
        lambda tokenizer, item, **kwargs: ("rendered prompt", 5),
    )

    hard, soft, _ = trainer._evaluate_prefix(
        "livemathematicianbench",
        FakePrefixModel(),
        [
            {
                "id": "math0",
                "question": "What is 1+1?",
                "choices": [
                    {"label": "A", "text": "1"},
                    {"label": "B", "text": "2"},
                ],
                "correct_choice": {"label": "B", "text": "2"},
            }
        ],
        cfg={},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "inference_backend": "vllm_prompt_embeds",
                "inference_base_url": "http://127.0.0.1:8010",
                "inference_timeout_seconds": 12.5,
                "injection_position": "skill_section",
                "max_prompt_tokens": 123,
                "max_new_tokens": 7,
            }
        ),
        out_dir=str(tmp_path / "prefix-livemath"),
        desc="  Val",
    )

    assert (hard, soft) == (1.0, 1.0)
    assert captured["base_url"] == "http://127.0.0.1:8010"
    assert captured["timeout_seconds"] == 12.5
    assert captured["prefix_rows"] == 4
    assert captured["prompts"] == ["rendered prompt"]
    assert captured["prefix_insert_indices"] == [5]
    assert captured["limits"] == (123, 7, 0.0)


def test_prefix_officeqa_uses_vllm_generator_with_prefix(monkeypatch) -> None:
    import torch

    from skillopt.softprefix import trainer
    from skillopt.softprefix.trainer import SoftPrefixSettings

    class FakeTokenizer:
        pass

    class FakePrefixModel:
        tokenizer = FakeTokenizer()
        prefix_embeddings = torch.zeros(4, 8)

        def generate_from_prompt(self, *args, **kwargs):
            raise AssertionError("prefix OfficeQA eval should use the vLLM generator")

    captured: dict[str, object] = {}

    class FakeVllmClient:
        def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
            captured["base_url"] = base_url
            captured["timeout_seconds"] = timeout_seconds

        def set_prefix(self, prefix_embeddings, *, injection_position=None) -> None:
            captured["prefix_rows"] = len(prefix_embeddings)
            captured["injection_position"] = injection_position

        def generate_from_prompts(self, prompts, *, max_prompt_tokens, max_new_tokens, temperature):
            captured["prompts"] = list(prompts)
            return ["<answer>gold</answer>"]

    monkeypatch.setattr("skillopt.softprefix.vllm_prompt_embeds.SoftPrefixVllmClient", FakeVllmClient)
    monkeypatch.setattr(trainer, "build_officeqa_prompt", lambda tokenizer, item, enable_thinking=False: "office prompt")

    hard, soft, _ = trainer._evaluate_prefix(
        "officeqa",
        FakePrefixModel(),
        [{"id": "uid0", "ground_truth": "gold"}],
        cfg={},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "inference_backend": "vllm_prompt_embeds",
                "inference_base_url": "http://127.0.0.1:8010",
            }
        ),
        out_dir="/tmp/prefix-officeqa",
        desc="  Val",
    )

    assert (hard, soft) == (1.0, 1.0)
    assert captured["prefix_rows"] == 4
    assert captured["prompts"] == ["office prompt"]


def test_prefix_officeqa_skill_section_uses_insert_indices(monkeypatch) -> None:
    from skillopt.softprefix import trainer
    from skillopt.softprefix.trainer import SoftPrefixSettings

    class FakeTokenizer:
        pass

    class FakePrefixModel:
        tokenizer = FakeTokenizer()
        prefix_embeddings = [[0.0] * 8 for _ in range(4)]

        def generate_from_prompt(self, *args, **kwargs):
            raise AssertionError("skill-section OfficeQA eval should use the vLLM generator")

    captured: dict[str, object] = {}

    class FakeVllmClient:
        def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
            captured["base_url"] = base_url
            captured["timeout_seconds"] = timeout_seconds

        def set_prefix(self, prefix_embeddings, *, injection_position=None) -> None:
            captured["prefix_rows"] = len(prefix_embeddings)
            captured["injection_position"] = injection_position

        def generate_from_prompts(
            self,
            prompts,
            *,
            max_prompt_tokens,
            max_new_tokens,
            temperature,
            prefix_insert_indices=None,
        ):
            captured["prompts"] = list(prompts)
            captured["prefix_insert_indices"] = list(prefix_insert_indices or [])
            captured["limits"] = (max_prompt_tokens, max_new_tokens, temperature)
            return ["<answer>gold</answer>"]

    monkeypatch.setattr("skillopt.softprefix.vllm_prompt_embeds.SoftPrefixVllmClient", FakeVllmClient)
    monkeypatch.setattr(
        trainer,
        "build_officeqa_prompt_and_insert_idx",
        lambda tokenizer, item, **kwargs: ("office prompt", 6),
    )

    hard, soft, _ = trainer._evaluate_prefix(
        "officeqa",
        FakePrefixModel(),
        [{"id": "uid0", "ground_truth": "gold"}],
        cfg={},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "inference_backend": "vllm_prompt_embeds",
                "inference_base_url": "http://127.0.0.1:8010",
                "injection_position": "skill_section",
                "max_prompt_tokens": 123,
                "max_new_tokens": 7,
            }
        ),
        out_dir="/tmp/prefix-officeqa-skill-section",
        desc="  Val",
    )

    assert (hard, soft) == (1.0, 1.0)
    assert captured["prefix_rows"] == 4
    assert captured["injection_position"] == "skill_section"
    assert captured["prompts"] == ["office prompt"]
    assert captured["prefix_insert_indices"] == [6]
    assert captured["limits"] == (123, 7, 0.0)


def test_prefix_officeqa_eval_can_use_local_tools(monkeypatch, tmp_path) -> None:
    import torch

    from skillopt.softprefix import trainer
    from skillopt.softprefix.trainer import SoftPrefixSettings

    class FakeTokenizer:
        pass

    class FakePrefixModel:
        tokenizer = FakeTokenizer()
        prefix_embeddings = torch.zeros(4, 8)

        def generate_from_prompt(self, *args, **kwargs):
            raise AssertionError("local-tool OfficeQA eval should use chat completions")

    class FakeVllmClient:
        def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
            self.base_url = base_url
            self.timeout_seconds = timeout_seconds
            self.calls = 0
            self.prefix_rows = None

        def set_prefix(self, prefix_embeddings, *, injection_position=None) -> None:
            self.prefix_rows = len(prefix_embeddings)
            self.injection_position = injection_position

        def generate_chat_completion(self, messages, **kwargs):
            self.calls += 1
            assert kwargs["tools"][0]["function"]["name"] == "glob"
            assert kwargs["tool_choice"] == "auto"
            assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
            if self.calls == 1:
                return (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read", "arguments": "{\"path\": \"doc.txt\"}"},
                            }
                        ],
                    },
                    {"finish_reason": "tool_calls"},
                )
            assert any(message.get("role") == "tool" and "needle evidence" in message.get("content", "") for message in messages)
            return {"role": "assistant", "content": "<answer>gold</answer>"}, {"finish_reason": "stop"}

    monkeypatch.setattr("skillopt.softprefix.vllm_prompt_embeds.SoftPrefixVllmClient", FakeVllmClient)
    monkeypatch.setattr(trainer, "resolve_docs_roots", lambda data_dirs=None: [str(tmp_path)])
    monkeypatch.setattr(trainer, "resolve_candidate_files", lambda source_files, docs_roots: [str(tmp_path / "doc.txt")])
    monkeypatch.setattr(trainer, "build_oracle_parsed_pages_context", lambda *args, **kwargs: "")
    monkeypatch.setattr(trainer, "run_officeqa_tool", lambda name, arguments, **kwargs: ("read doc.txt", "needle evidence"))

    hard, soft, rows = trainer._evaluate_prefix(
        "officeqa",
        FakePrefixModel(),
        [{"id": "uid0", "question": "q", "ground_truth": "gold", "source_files": ["doc.txt"]}],
        cfg={"use_local_tools": True, "max_tool_turns": 3, "data_dirs": [str(tmp_path)]},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "inference_backend": "vllm_prompt_embeds",
                "inference_base_url": "http://127.0.0.1:8010",
            }
        ),
        out_dir=str(tmp_path / "eval"),
        desc="  Val",
    )

    assert (hard, soft) == (1.0, 1.0)
    assert rows[0]["use_local_tools"] is True
    assert rows[0]["n_turns"] == 4


def test_prefix_officeqa_eval_accepts_answer_tool_call(monkeypatch, tmp_path) -> None:
    from skillopt.softprefix import trainer

    class FakeGenerator:
        def generate_chat_completion(self, messages, **kwargs):
            return (
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_answer",
                            "type": "function",
                            "function": {
                                "name": "answer",
                                "arguments": "{\"answer\": \"gold\"}",
                            },
                        }
                    ],
                },
                {"finish_reason": "tool_calls"},
            )

    monkeypatch.setattr(trainer, "resolve_docs_roots", lambda data_dirs=None: [str(tmp_path)])
    monkeypatch.setattr(trainer, "resolve_candidate_files", lambda source_files, docs_roots: [str(tmp_path / "doc.txt")])
    monkeypatch.setattr(trainer, "build_oracle_parsed_pages_context", lambda *args, **kwargs: "")

    hard, soft, rows = trainer._evaluate_officeqa_prefix_with_local_tools(
        [{"id": "uid0", "question": "q", "ground_truth": "gold", "source_files": ["doc.txt"]}],
        out_dir=str(tmp_path / "eval"),
        max_prompt_tokens=128,
        max_new_tokens=32,
        temperature=0.0,
        desc="  Val",
        generator=FakeGenerator(),
        max_tool_turns=3,
        data_dirs=[str(tmp_path)],
        search_mode="offline",
        injection_position="prompt_start",
    )

    assert (hard, soft) == (1.0, 1.0)
    assert rows[0]["predicted_answer"] == "gold"


def test_prefix_spreadsheet_eval_repairs_execution_error(monkeypatch, tmp_path) -> None:
    from skillopt.softprefix import trainer

    class FakePrefixModel:
        tokenizer = object()

    class FakeGenerator:
        def __init__(self) -> None:
            self.messages = []

        def generate_from_messages(self, messages, **kwargs):
            self.messages.append([dict(message) for message in messages])
            if len(self.messages) == 1:
                return "```python\nraise SyntaxError('bad')\n```"
            assert any("raised an error" in message.get("content", "") for message in messages)
            return "```python\n# fixed\n```"

    exec_results = iter([(False, "SyntaxError: bad"), (True, ""), (True, "")])

    monkeypatch.setattr(trainer, "find_spreadsheet_test_cases", lambda task_dir: [("1", "input.xlsx", "gold.xlsx")])
    monkeypatch.setattr(trainer, "build_spreadsheet_system", lambda skill_content: "system prompt")
    monkeypatch.setattr(trainer, "build_spreadsheet_user", lambda *args, **kwargs: "user prompt")
    monkeypatch.setattr(
        trainer,
        "build_spreadsheet_codegen_prompt_and_insert_idx",
        lambda *args, **kwargs: ("rendered prompt", None),
    )
    monkeypatch.setattr(trainer, "run_spreadsheet_generated_code", lambda *args, **kwargs: next(exec_results))
    monkeypatch.setattr(trainer, "evaluate_spreadsheet", lambda *args, **kwargs: {"ok": True, "reason": ""})
    monkeypatch.setattr(trainer, "auto_verify_spreadsheet_output", lambda *args, **kwargs: "verified")

    generator = FakeGenerator()
    hard, soft, rows = trainer.evaluate_spreadsheet_prefix(
        FakePrefixModel(),
        [
            {
                "id": "sheet0",
                "instruction": "do it",
                "instruction_type": "Cell-Level Manipulation",
                "answer_position": "A1",
                "spreadsheet_path": "sheet0",
            }
        ],
        out_dir=str(tmp_path / "eval"),
        data_root=str(tmp_path),
        max_prompt_tokens=128,
        max_new_tokens=32,
        temperature=0.0,
        generator=generator,
        repair_turns=2,
    )

    assert (hard, soft) == (1.0, 1.0)
    assert rows[0]["n_turns"] == 2
    assert len(generator.messages) == 2


def test_docvqa_eval_uses_generator_messages(monkeypatch, tmp_path) -> None:
    from skillopt.softprefix import trainer

    class FakeModel:
        def generate_from_messages(self, *args, **kwargs):
            raise AssertionError("DocVQA eval should use the vLLM message generator")

    class FakeGenerator:
        def __init__(self) -> None:
            self.messages = None

        def generate_from_messages(self, messages, *, max_prompt_tokens, max_new_tokens, temperature, max_image_tokens=0):
            self.messages = messages
            assert max_prompt_tokens == 128
            assert max_new_tokens == 8
            assert temperature == 0.0
            assert max_image_tokens == 64
            return "invoice total"

    messages = [{"role": "user", "content": [{"type": "text", "text": "What is shown?"}]}]
    generator = FakeGenerator()
    monkeypatch.setattr(trainer, "build_docvqa_messages", lambda item, image_detail="auto": messages)

    hard, soft, _ = trainer.evaluate_docvqa_prefix(
        FakeModel(),
        [{"id": "doc0", "question": "q", "answers": ["invoice total"], "image_path": "/tmp/doc.png"}],
        out_dir=str(tmp_path),
        max_prompt_tokens=128,
        max_new_tokens=8,
        temperature=0.0,
        max_image_tokens=64,
        generator=generator,
    )

    assert (hard, soft) == (1.0, 1.0)
    assert generator.messages == messages


def test_softprefix_server_exposes_openai_models_endpoint() -> None:
    from http.server import ThreadingHTTPServer

    from skillopt.softprefix.vllm_prompt_embeds import _Handler

    class FakeEngine:
        model_name = "Qwen/Qwen3.5-4B"

    _Handler.engine = FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        data = _read_http_json(f"http://{host}:{port}/v1/models")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert data["object"] == "list"
    assert data["data"][0]["id"] == "Qwen/Qwen3.5-4B"


def test_softprefix_server_exposes_openai_chat_completions_endpoint() -> None:
    from http.server import ThreadingHTTPServer

    from skillopt.softprefix.vllm_prompt_embeds import _Handler

    class FakeEngine:
        model_name = "Qwen/Qwen3.5-4B"

        def generate_plain_from_messages(self, messages, *, max_new_tokens, temperature):
            assert messages == [{"role": "user", "content": "ping"}]
            assert max_new_tokens == 7
            assert temperature == 0.25
            return "pong"

    _Handler.engine = FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        data = _read_http_json(
            f"http://{host}:{port}/v1/chat/completions",
            payload={
                "model": "Qwen/Qwen3.5-4B",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 7,
                "temperature": 0.25,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert data["object"] == "chat.completion"
    assert data["model"] == "Qwen/Qwen3.5-4B"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "pong"}


def test_softprefix_server_can_use_prefix_for_chat_completions() -> None:
    from http.server import ThreadingHTTPServer

    from skillopt.softprefix.vllm_prompt_embeds import _Handler

    class FakeEngine:
        model_name = "Qwen/Qwen3.5-4B"

        def generate_from_messages(
            self,
            messages,
            *,
            max_prompt_tokens,
            max_new_tokens,
            temperature,
            tools=None,
            tool_choice=None,
            chat_template_kwargs=None,
        ):
            assert messages == [{"role": "user", "content": "ping"}]
            assert max_prompt_tokens == 123
            assert max_new_tokens == 7
            assert temperature == 0.25
            assert tools == [{"type": "function", "function": {"name": "read"}}]
            assert tool_choice == "auto"
            assert chat_template_kwargs == {"enable_thinking": False}
            return "pong with prefix"

    _Handler.engine = FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        data = _read_http_json(
            f"http://{host}:{port}/v1/chat/completions",
            payload={
                "model": "Qwen/Qwen3.5-4B",
                "messages": [{"role": "user", "content": "ping"}],
                "max_prompt_tokens": 123,
                "max_tokens": 7,
                "temperature": 0.25,
                "tools": [{"type": "function", "function": {"name": "read"}}],
                "tool_choice": "auto",
                "chat_template_kwargs": {"enable_thinking": False},
                "use_prefix": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert data["choices"][0]["message"] == {"role": "assistant", "content": "pong with prefix"}


def test_softprefix_server_parses_qwen_tool_calls() -> None:
    from http.server import ThreadingHTTPServer

    from skillopt.softprefix.vllm_prompt_embeds import _Handler

    class FakeEngine:
        model_name = "Qwen/Qwen3.5-4B"
        enable_auto_tool_choice = True
        tool_call_parser = "qwen3_coder"

        def generate_plain_from_messages(
            self,
            messages,
            *,
            max_new_tokens,
            temperature,
            tools=None,
            tool_choice=None,
            chat_template_kwargs=None,
        ):
            assert messages == [{"role": "user", "content": "find it"}]
            assert max_new_tokens == 16
            assert temperature == 0.0
            assert tool_choice == "auto"
            assert chat_template_kwargs == {"enable_thinking": False}
            assert tools and tools[0]["function"]["name"] == "read_file"
            return (
                "I will inspect the file.\n"
                "<tool_call>\n"
                "<function=read_file>\n"
                "<parameter=path>\n"
                "README.md\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>"
            )

    _Handler.engine = FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        data = _read_http_json(
            f"http://{host}:{port}/v1/chat/completions",
            payload={
                "model": "Qwen/Qwen3.5-4B",
                "messages": [{"role": "user", "content": "find it"}],
                "max_tokens": 16,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"] == {
        "role": "assistant",
        "content": "I will inspect the file.\n",
        "tool_calls": [
            {
                "id": "call_softprefix_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
            }
        ],
    }


def test_softprefix_qwen_tool_parser_ignores_unknown_non_answer_tools() -> None:
    vllm_prompt_embeds = _load_vllm_prompt_embeds_module()

    text = (
        "Need another action.\n"
        "<tool_call>\n"
        "<function=lookup>\n"
        "<parameter=query>\n"
        "needle\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    content, tool_calls = vllm_prompt_embeds._parse_qwen_tool_calls(
        text,
        [{"type": "function", "function": {"name": "read"}}],
    )

    assert content == text
    assert tool_calls == []


def test_softprefix_main_accepts_qwen3_reasoning_parser(monkeypatch) -> None:
    vllm_prompt_embeds = _load_vllm_prompt_embeds_module()

    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, address, handler_class):
            captured["address"] = address
            captured["handler_class"] = handler_class

        def serve_forever(self):
            captured["served"] = True

    def fake_engine(**kwargs):
        captured["engine_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(vllm_prompt_embeds, "SoftPrefixVllmEngine", fake_engine)
    monkeypatch.setattr(vllm_prompt_embeds, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vllm_prompt_embeds",
            "--model_name",
            "Qwen/Qwen3.5-4B",
            "--reasoning-parser",
            "qwen3",
        ],
    )

    vllm_prompt_embeds.main()

    assert captured["engine_kwargs"]["reasoning_parser"] == "qwen3"
    assert captured["served"] is True


def test_softprefix_server_parses_qwen3_reasoning_content() -> None:
    from http.server import ThreadingHTTPServer

    vllm_prompt_embeds = _load_vllm_prompt_embeds_module()
    _Handler = vllm_prompt_embeds._Handler

    class FakeEngine:
        model_name = "Qwen/Qwen3.5-4B"
        enable_auto_tool_choice = False
        tool_call_parser = ""
        reasoning_parser = "qwen3"

        def generate_plain_from_messages(
            self,
            messages,
            *,
            max_new_tokens,
            temperature,
            tools=None,
            tool_choice=None,
            chat_template_kwargs=None,
        ):
            del messages, max_new_tokens, temperature, tools, tool_choice, chat_template_kwargs
            return "<think>\nI should answer briefly.\n</think>\nThe answer is 42."

    _Handler.engine = FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        data = _read_http_json(
            f"http://{host}:{port}/v1/chat/completions",
            payload={
                "model": "Qwen/Qwen3.5-4B",
                "messages": [{"role": "user", "content": "answer"}],
                "max_tokens": 16,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert data["choices"][0]["message"] == {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "I should answer briefly.",
    }
