"""ALFWorld rollout module for ReflACT.

Provides:
  - build_alfworld_env(): build ALFWorld environment (wraps vendored SkillRL env)
  - run_alfworld_batch(): run a batch of ALFWorld episodes in parallel
  - TASKS: list of ALFWorld task types
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
from typing import Any, Callable

from tqdm import tqdm

from skillopt.model import chat_target

SYSTEM_PROMPT = "You are an expert agent operating in the ALFRED Embodied Environment."

# ── Constants ─────────────────────────────────────────────────────────────────

TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_task_type(gamefile: str) -> str:
    for task in TASKS:
        if task in gamefile:
            return task
    return "other"


def _extract_action(model_response: str) -> str | None:
    match = re.search(r"<action>(.*?)</action>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_think(model_response: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def _fallback_or_raise_invalid_response(
    response: str,
    *,
    task_id: str,
    step_idx: int,
    reason: str,
    fallback_on_invalid_response: bool,
) -> str:
    raw_response = (response or "").strip()
    print(
        f"    [alfworld rollout] invalid model response "
        f"id={task_id} step={step_idx} reason={reason!r}\n"
        f"    raw_response={raw_response!r}",
        flush=True,
    )
    if fallback_on_invalid_response:
        return f"<think>{reason}</think><action>look</action>"
    snippet = raw_response.replace("\n", "\\n")
    if len(snippet) > 500:
        snippet = snippet[:500] + "..."
    raise ValueError(
        f"ALFWorld rollout model response {reason} "
        f"id={task_id} step={step_idx}; raw_response={snippet!r}"
    )


def _invalid_response_reason(response: str) -> str | None:
    if not (response or "").strip():
        return "empty model response"
    if _extract_action(response) is None:
        return "missing <action> tag"
    return None


def _resolve_model_response_with_retries(
    fetch_response: Callable[[], str],
    *,
    task_id: str,
    step_idx: int,
    invalid_response_retries: int,
    fallback_on_invalid_response: bool,
) -> str:
    """Fetch a model response, retrying when no action can be extracted."""
    max_attempts = max(1, int(invalid_response_retries) + 1)
    last_response = ""
    reason = "empty model response"
    for attempt in range(1, max_attempts + 1):
        last_response = (fetch_response() or "").strip()
        reason = _invalid_response_reason(last_response)
        if reason is None:
            return last_response
        if attempt < max_attempts:
            print(
                f"    [alfworld rollout] retrying invalid model response "
                f"id={task_id} step={step_idx} reason={reason!r} "
                f"attempt={attempt}/{max_attempts}",
                flush=True,
            )
    return _fallback_or_raise_invalid_response(
        last_response,
        task_id=task_id,
        step_idx=step_idx,
        reason=reason,
        fallback_on_invalid_response=fallback_on_invalid_response,
    )


def _build_skill_prompt(skill_content: str) -> str:
    """Build the skill section to inject into the agent's system prompt."""
    if not skill_content or not skill_content.strip():
        return ""
    return (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{skill_content}\n"
    )


def _append_diagnostic_instruction(prompt: str, diagnostic_instruction: str) -> str:
    if not diagnostic_instruction or not diagnostic_instruction.strip():
        return prompt
    return f"{prompt}\n\n## Training Readout\n{diagnostic_instruction.strip()}\n"


def _get_result_id(idx: int, result_ids: list[str] | None = None) -> str:
    return str(result_ids[idx]) if result_ids and idx < len(result_ids) else f"env_{idx:03d}"


def _save_conversation(pred_dir: str, task_id: str, conversation: list[dict]) -> None:
    if not pred_dir:
        return
    conv_dir = os.path.join(pred_dir, task_id)
    os.makedirs(conv_dir, exist_ok=True)
    with open(os.path.join(conv_dir, "conversation.json"), "w") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)


def _cached_model_response(record: dict) -> str:
    response = str(record.get("model_response") or "").strip()
    if response:
        return response
    action = str(record.get("action") or "").strip()
    return f"<action>{action}</action>" if action else "None"


def _cached_step_records(conversation: list[dict], max_steps: int) -> list[dict]:
    records = [
        record
        for record in conversation
        if int(record.get("step", -1)) >= 0
        and int(record.get("step", -1)) < max_steps
        and (record.get("model_response") or record.get("action"))
    ]
    return sorted(records, key=lambda record: int(record.get("step", -1)))


# ── Environment builder ──────────────────────────────────────────────────────


def build_alfworld_env(
    env_num: int,
    eval_dataset: str = "eval_out_of_distribution",
    seed: int = 42,
    is_train: bool = False,
    specific_gamefiles: list[str] | None = None,
):
    """Build ALFWorld environment manager.

    Args:
        env_num: number of parallel environments
        eval_dataset: 'eval_in_distribution' or 'eval_out_of_distribution' or train
        seed: random seed
        is_train: whether to use training set

    Returns:
        env_manager: AlfWorldEnvironmentManager instance
    """
    from functools import partial

    from omegaconf import OmegaConf

    from skillopt.envs.alfworld.vendor.alfworld_envs import build_alfworld_envs
    from skillopt.envs.alfworld.vendor.alfworld_projection import alfworld_projection
    from skillopt.envs.alfworld.vendor.env_manager import AlfWorldEnvironmentManager

    HERE = os.path.dirname(os.path.abspath(__file__))

    alf_config_path = os.path.join(HERE, "vendor", "config_tw.yaml")
    env_kwargs = {"eval_dataset": eval_dataset}

    envs = build_alfworld_envs(
        alf_config_path,
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=is_train,
        env_kwargs=env_kwargs,
        resources_per_worker=None,
        gamefiles=specific_gamefiles,
    )

    config = OmegaConf.create(
        {
            "env": {
                "history_length": 2,
                "env_name": "alfworld/AlfredTWEnv",
            }
        }
    )

    projection_f = partial(alfworld_projection)
    env_manager = AlfWorldEnvironmentManager(envs, projection_f, config)
    return env_manager


# ── Batch rollout ─────────────────────────────────────────────────────────────


def run_alfworld_batch(
    env_manager,
    skill_content: str,
    max_steps: int = 50,
    out_root: str = "",
    max_api_workers: int = 8,
    temperature: float = 0.4,
    max_completion_tokens: int = 16384,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    result_ids: list[str] | None = None,
    cached_conversations: dict[str, list[dict]] | None = None,
    generator: Any | None = None,
    prompt_renderer: Callable[[str, str], str] | None = None,
    max_prompt_tokens: int = 16384,
    api_timeout_seconds: float = 30.0,
    fallback_on_invalid_response: bool = True,
    invalid_response_retries: int = 3,
) -> list[dict]:
    """Run a batch of ALFWorld episodes.

    Returns a list of result dicts compatible with SkillOpt pipeline:
    [
        {
            "id": "<env_idx>_<gamefile_hash>",
            "hard": 0 or 1,
            "soft": 0.0 or 1.0,
            "n_turns": <int>,
            "fail_reason": "<str>",
            "agent_ok": True,
            "task_type": "<str>",
            "gamefile": "<str>",
            "task_description": "<str>",
        },
        ...
    ]

    Also saves conversation.json per environment in out_root/predictions/<task_id>/ as steps complete.
    """
    skill_prompt = _build_skill_prompt(skill_content)

    obs, infos = env_manager.reset({})
    env_num = len(obs["text"])
    env_dones = [False] * env_num
    overall_success = [False] * env_num
    pred_dir = os.path.join(out_root, "predictions") if out_root else ""

    # Build per-env metadata
    env_meta: list[dict] = []
    for i in range(env_num):
        gamefile = infos[i].get("extra.gamefile", "") if isinstance(infos[i], dict) else ""
        task_type = _get_task_type(gamefile)
        # Extract task description from initial observation
        task_desc = ""
        anchor_text = obs["anchor"][i] if "anchor" in obs else ""
        task_start = anchor_text.find("Your task is to: ")
        if task_start != -1:
            task_desc = anchor_text[task_start + len("Your task is to: "):].strip()

        env_meta.append({
            "gamefile": gamefile,
            "task_type": task_type,
            "task_description": task_desc,
        })

    # Per-env conversation records — seed each with the initial observation
    conversations: list[list[dict]] = [[] for _ in range(env_num)]
    cached_records_by_env: list[list[dict]] = []
    for i in range(env_num):
        task_id = _get_result_id(i, result_ids)
        cached = cached_conversations.get(task_id) if cached_conversations else None
        if cached is not None:
            conversations[i] = [dict(record) for record in cached]
            cached_records_by_env.append(_cached_step_records(conversations[i], max_steps))
        else:
            conversations[i].append({
                "step": -1,
                "type": "initial_obs",
                "obs_anchor": obs["anchor"][i] if "anchor" in obs else "",
                "obs_text": obs["text"][i],
            })
            cached_records_by_env.append([])
        _save_conversation(pred_dir, task_id, conversations[i])

    replay_count = 0
    if cached_conversations:
        replay_counts = {len(records) for records in cached_records_by_env}
        if len(replay_counts) != 1:
            raise ValueError("Cached ALFWorld conversations must have the same replay length within a batch")
        replay_count = replay_counts.pop()
        with tqdm(range(replay_count), desc="    ALFWorld replay", unit="step", leave=False) as replay_progress:
            for replay_idx in replay_progress:
                replay_actions = ["None"] * env_num
                for i in range(env_num):
                    replay_actions[i] = _cached_model_response(cached_records_by_env[i][replay_idx])
                obs, rewards, dones, infos = env_manager.step(replay_actions)
                for i in range(env_num):
                    if dones[i]:
                        env_dones[i] = True
                        overall_success[i] = bool(infos[i].get("won", False))
                replay_progress.set_postfix(
                    active=sum(1 for done in env_dones if not done),
                    done=sum(1 for done in env_dones if done),
                    success=sum(1 for won in overall_success if won),
                )

    with tqdm(range(replay_count, max_steps), desc="    ALFWorld rollout", unit="step", leave=False) as progress:
        for step_idx in progress:
            if all(env_dones):
                progress.set_postfix(active=0, done=env_num, success=sum(overall_success))
                break

            active_indices = [i for i in range(env_num) if not env_dones[i]]

            # Build prompts with skill injection
            prompts: dict[int, str] = {}
            for i in active_indices:
                prompt = obs["text"][i]
                if skill_prompt:
                    # Inject skill before the action instruction
                    prompt = skill_prompt + "\n" + prompt
                if diagnostic_mode and diagnostic_instruction.strip():
                    prompt = _append_diagnostic_instruction(prompt, diagnostic_instruction)
                prompts[i] = prompt

            actions = ["None"] * env_num

            if generator is not None:
                render_prompt = prompt_renderer or (lambda system, user: f"{system}\n{user}")
                rendered_prompts = {
                    idx: render_prompt(SYSTEM_PROMPT, prompts[idx]) for idx in active_indices
                }
                try:
                    batch_responses = generator.generate_from_prompts(
                        [rendered_prompts[idx] for idx in active_indices],
                        max_prompt_tokens=max_prompt_tokens,
                        max_new_tokens=max_completion_tokens,
                        temperature=temperature,
                    )
                except Exception as e:
                    if not fallback_on_invalid_response:
                        raise RuntimeError(
                            f"ALFWorld rollout generator error step={step_idx}: {type(e).__name__}: {e}"
                        ) from e
                    print(
                        f"    [alfworld rollout] generator error "
                        f"step={step_idx}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    batch_responses = [
                        "<think>error</think><action>look</action>"
                    ] * len(active_indices)

                for idx, initial_response in zip(active_indices, batch_responses):
                    task_id = _get_result_id(idx, result_ids)
                    prompt = rendered_prompts[idx]

                    def fetch_response(single_prompt: str = prompt) -> str:
                        return generator.generate_from_prompts(
                            [single_prompt],
                            max_prompt_tokens=max_prompt_tokens,
                            max_new_tokens=max_completion_tokens,
                            temperature=temperature,
                        )[0]

                    initial = (initial_response or "").strip()
                    if _invalid_response_reason(initial) is None:
                        actions[idx] = initial
                        continue

                    max_attempts = max(1, int(invalid_response_retries) + 1)
                    last_response = initial
                    reason = _invalid_response_reason(last_response) or "empty model response"
                    for attempt in range(2, max_attempts + 1):
                        print(
                            f"    [alfworld rollout] retrying invalid model response "
                            f"id={task_id} step={step_idx} reason={reason!r} "
                            f"attempt={attempt}/{max_attempts}",
                            flush=True,
                        )
                        last_response = (fetch_response() or "").strip()
                        reason = _invalid_response_reason(last_response)
                        if reason is None:
                            actions[idx] = last_response
                            break
                    else:
                        actions[idx] = _fallback_or_raise_invalid_response(
                            last_response,
                            task_id=task_id,
                            step_idx=step_idx,
                            reason=reason or "empty model response",
                            fallback_on_invalid_response=fallback_on_invalid_response,
                        )
            else:
                def call_api(idx):
                    task_id = _get_result_id(idx, result_ids)
                    try:
                        return idx, _resolve_model_response_with_retries(
                            lambda: chat_target(
                                system=SYSTEM_PROMPT,
                                user=prompts[idx],
                                max_completion_tokens=max_completion_tokens,
                                retries=5,
                                stage="rollout",
                                timeout=api_timeout_seconds,
                            )[0],
                            task_id=task_id,
                            step_idx=step_idx,
                            invalid_response_retries=invalid_response_retries,
                            fallback_on_invalid_response=fallback_on_invalid_response,
                        )
                    except Exception as e:
                        if not fallback_on_invalid_response:
                            raise
                        print(
                            f"    [alfworld rollout] target API error "
                            f"id={task_id} env={idx} step={step_idx}: {type(e).__name__}: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return idx, "<think>error</think><action>look</action>"

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_api_workers)
                try:
                    futures = {executor.submit(call_api, i): i for i in active_indices}
                    pending_futs = set(futures)
                    while pending_futs:
                        done, _ = concurrent.futures.wait(
                            pending_futs,
                            timeout=5,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            pending_futs.remove(future)
                            try:
                                idx, response = future.result()
                            except Exception:  # noqa: BLE001
                                if not fallback_on_invalid_response:
                                    raise
                                idx = futures[future]
                                response = "<think>error</think><action>look</action>"
                            actions[idx] = response
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

            # Save model responses before stepping
            model_responses = {i: actions[i] for i in active_indices}

            # Step environment
            obs, rewards, dones, infos = env_manager.step(actions)

            # Record trajectory
            for i in active_indices:
                step_record = {
                    "step": step_idx,
                    "prompt": prompts[i],  # exact obs text fed to the model at this step
                    "action": _extract_action(model_responses[i]),
                    "reasoning": _extract_think(model_responses[i]),
                    "model_response": model_responses[i],
                    "env_feedback": obs["anchor"][i] if "anchor" in obs else "",
                    "reward": float(rewards[i]),
                    "done": bool(dones[i]),
                }
                conversations[i].append(step_record)
                _save_conversation(pred_dir, _get_result_id(i, result_ids), conversations[i])

            # Update done status
            for i in range(env_num):
                if env_dones[i]:
                    continue
                if dones[i]:
                    env_dones[i] = True
                    won = bool(infos[i].get("won", False))
                    overall_success[i] = won
            progress.set_postfix(
                active=sum(1 for done in env_dones if not done),
                done=sum(1 for done in env_dones if done),
                success=sum(1 for won in overall_success if won),
            )

    # Build results and save conversations
    results: list[dict] = []

    for i in range(env_num):
        gamefile = env_meta[i]["gamefile"]
        task_type = env_meta[i]["task_type"]
        task_desc = env_meta[i]["task_description"]
        n_turns = len(conversations[i])
        won = overall_success[i]

        # Generate stable task ID from env index and gamefile
        task_id = _get_result_id(i, result_ids)

        fail_reason = ""
        if not won:
            if not env_dones[i]:
                fail_reason = f"Timeout after {max_steps} steps"
            else:
                fail_reason = "Episode ended without completing the task"

        result = {
            "id": task_id,
            "hard": 1 if won else 0,
            "soft": 1.0 if won else 0.0,
            "n_turns": n_turns,
            "fail_reason": fail_reason,
            "agent_ok": True,  # ALFWorld agent always runs OK (no crash)
            "task_type": task_type,
            "gamefile": gamefile,
            "task_description": task_desc,
            "instruction_type": task_type,  # for compatibility with v2 pipeline
        }
        results.append(result)

        # Save conversation
        _save_conversation(pred_dir, task_id, conversations[i])

    return results


# ── Item loading (for compatibility with split_three_way) ────────────────────


def load_alfworld_items(
    eval_dataset: str,
    env_num: int,
    seed: int = 42,
    is_train: bool = False,
) -> list[dict]:
    """Create pseudo-item dicts for ALFWorld environments.

    Since ALFWorld doesn't have a static JSON dataset like SpreadsheetBench,
    we create lightweight item dicts that carry enough metadata for the pipeline.
    The actual environment is built dynamically.

    Returns:
        List of dicts with "id" keys, one per environment slot.
    """
    items = []
    for i in range(env_num):
        items.append({
            "id": f"env_{i:03d}",
            "eval_dataset": eval_dataset,
            "env_index": i,
        })
    return items
