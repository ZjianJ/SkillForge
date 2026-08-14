from __future__ import annotations

import numpy as np
import torch

from skillopt.softprefix.prcb import (
    causal_prefix_pair,
    choose_prefix_pair,
    mask_prefix_gradient_,
    margin_decision_locator,
    reference_decision_margin,
    residual_combined_scores,
    select_harmful_anchor_positions,
    select_margin_decision_top_fraction,
    select_positive_top_fraction,
    shrink_pair_update_,
    sliding_prefix_pair,
    topk_residual_forward_kl,
    topk_residual_js,
)
from scripts.train_spreadsheetbench_prcb_v1 import (
    fixed_monitor_split,
    optimizer_steps_by_round,
    parse_stage_alpha_grid,
    round_schedule,
)
from scripts.train_spreadsheetbench_prcb_v6 import build_stage_rows
from skillopt.softprefix.prcb_v6 import (
    centered_logit_delta,
    choose_stage_alpha,
    combine_boosted_logits,
    topk_residual_kl_from_logits,
    topk_reference_from_logits,
)


def test_stage_alpha_grid_supports_rejection_and_deduplicates() -> None:
    assert parse_stage_alpha_grid("0.5,0,0.25,0.25") == [0.0, 0.25, 0.5]


def test_v6_logit_ensemble_is_additive_and_shift_invariant() -> None:
    base = torch.tensor([[1.0, 2.0, 4.0]])
    first = torch.tensor([[3.0, 4.0, 6.0]])  # Same distribution: constant +2.
    second = torch.tensor([[1.0, 4.0, 2.0]])
    assert torch.count_nonzero(centered_logit_delta(torch, first, base)) == 0
    combined = combine_boosted_logits(torch, base, [first, second], [1.0, 0.5])
    expected_delta = centered_logit_delta(torch, second, base) * 0.5
    torch.testing.assert_close(combined, base + expected_delta)


def test_v6_topk_kl_and_alpha_rejection() -> None:
    logits = torch.tensor([[3.0, 1.0, 0.0, -1.0]])
    ids, logp, residual = topk_reference_from_logits(torch, logits, top_k=2)
    loss = topk_residual_kl_from_logits(
        torch,
        logits,
        reference_topk_ids=ids,
        reference_topk_logp=logp,
        reference_residual_log_mass=residual,
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    chosen = choose_stage_alpha(
        [
            {"alpha": 0.0, "global_loss": 1.0, "safe": True},
            {"alpha": 0.5, "global_loss": 1.1, "safe": True},
        ]
    )
    assert chosen["alpha"] == 0.0


def test_v6_topk_kl_renormalizes_low_precision_reference() -> None:
    logits = torch.tensor([[3.0, 1.0, -1.0, -2.0]])
    ids, logp, residual = topk_reference_from_logits(torch, logits, top_k=2)
    loss = topk_residual_kl_from_logits(
        torch,
        logits,
        reference_topk_ids=ids,
        reference_topk_logp=logp.to(torch.float16),
        reference_residual_log_mass=residual.to(torch.float16),
    )
    assert float(loss) >= 0.0
    assert float(loss) < 1e-6


def test_v6_current_core_is_removed_from_history_replay() -> None:
    rows = [
        {
            "id": "example",
            "messages": [],
            "target": "x",
            "score_cache": "unused.npz",
        }
    ]
    scores = {
        "example": {
            "target_ids": np.arange(7),
            "dynamic_score": np.array([10.0, 9.0, 8.0, 7.0, 0.0, 0.0, 0.0]),
            "clean_kl": np.array([0.0, 0.0, 0.0, 0.0, 4.0, 3.0, 0.0]),
            "skill_benefit": np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        }
    }
    built, _, _ = build_stage_rows(
        rows,
        scores,
        {"example": {0, 4}},
        ratio=0.25,
    )
    assert built[0]["core_indices"] == [0, 1]
    assert built[0]["history_indices"] == [4]
    assert set(built[0]["core_indices"]).isdisjoint(built[0]["history_indices"])
    assert set(built[0]["anchor_indices"]).isdisjoint(
        set(built[0]["core_indices"]) | set(built[0]["history_indices"])
    )


def test_margin_locator_prioritizes_unresolved_skill_decision_flips() -> None:
    targets = np.array([1, 2, 3])
    skill_ids = np.array([[1, 0], [4, 2], [3, 5]])
    clean_ids = np.array([[0, 1], [4, 2], [3, 5]])
    skill_logp = np.array([[-0.1, -1.1], [-0.2, -0.4], [-0.1, -2.1]])
    clean_logp = np.array([[-0.2, -0.7], [-0.1, -0.5], [-0.1, -1.1]])
    skill_margin, skill_top1 = reference_decision_margin(
        target_ids=targets,
        target_logp=np.array([-0.1, -0.4, -0.1]),
        topk_ids=skill_ids,
        topk_logp=skill_logp,
    )
    clean_margin, clean_top1 = reference_decision_margin(
        target_ids=targets,
        target_logp=np.array([-0.7, -0.5, -0.1]),
        topk_ids=clean_ids,
        topk_logp=clean_logp,
    )
    values = margin_decision_locator(
        skill_margin=skill_margin,
        clean_margin=clean_margin,
        current_margin=np.array([-0.8, -0.3, 2.5]),
        skill_top1_gold=skill_top1,
        clean_top1_gold=clean_top1,
        current_top1_gold=np.array([False, False, True]),
    )
    assert values["margin_priority"].tolist() == [2, 1, 0]
    # Tier 2 wins even if Tier 1 has the larger numerical residual.
    selected = select_margin_decision_top_fraction(
        priority=values["margin_priority"],
        residual=np.array([0.2, 9.0, 0.0]),
        original_gain=values["original_margin_gain"],
        teacher_student_js=np.array([0.01, 0.9, 1.0]),
        ratio=0.2,
    )
    assert selected == [0]


def test_causal_prefix_pair_covers_all_rows_in_both_directions():
    tail_first = [
        causal_prefix_pair(prefix_length=8, round_index=index, direction="tail_to_head")
        for index in range(1, 5)
    ]
    head_first = [
        causal_prefix_pair(prefix_length=8, round_index=index, direction="head_to_tail")
        for index in range(1, 5)
    ]
    assert tail_first == [[6, 7], [4, 5], [2, 3], [0, 1]]
    assert head_first == list(reversed(tail_first))
    assert sorted(index for pair in tail_first for index in pair) == list(range(8))


def test_sliding_overlap_one_schedule_and_budget_match_v3() -> None:
    pairs = [
        sliding_prefix_pair(prefix_length=8, round_index=index)
        for index in range(1, 8)
    ]
    assert pairs == [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
    steps = optimizer_steps_by_round(
        rounds=7,
        steps_per_round=8,
        pattern="5,4,5,4,5,4,5",
    )
    assert sum(steps) == 32
    schedules = []
    offset = 0
    for round_index, round_steps in enumerate(steps, start=1):
        examples = round_steps * 2
        schedules.extend(
            round_schedule(
                61,
                round_index=round_index,
                examples=examples,
                seed=1,
                start_offset=offset,
            )
        )
        offset += examples
    assert len(schedules) == 64
    assert len(set(schedules)) == 61


def test_sliding_window_five_complete_schedule() -> None:
    windows = [
        sliding_prefix_pair(prefix_length=8, round_index=index, window_size=5)
        for index in range(1, 5)
    ]
    assert windows == [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
    ]


def test_fixed_monitor_split_is_deterministic_and_disjoint() -> None:
    rows = [{"id": str(index)} for index in range(61)]
    first_train, first_monitor = fixed_monitor_split(
        rows, monitor_count=12, seed=1
    )
    second_train, second_monitor = fixed_monitor_split(
        rows, monitor_count=12, seed=1
    )
    assert (first_train, first_monitor) == (second_train, second_monitor)
    assert len(first_train) == 49
    assert len(first_monitor) == 12
    assert set(first_train).isdisjoint(first_monitor)
    assert set(first_train) | set(first_monitor) == {str(index) for index in range(61)}


def test_topk_residual_divergences_are_zero_for_matching_distribution() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    logp = torch.log_softmax(logits, dim=-1)
    ids = torch.tensor([[0, 1]])
    ref_top = logp.gather(-1, ids)
    ref_residual = torch.log1p(-ref_top.exp().sum(dim=-1))
    js = topk_residual_js(
        torch,
        logits,
        reference_topk_ids=ids,
        reference_topk_logp=ref_top,
        reference_residual_log_mass=ref_residual,
    )
    kl = topk_residual_forward_kl(
        torch,
        logits,
        reference_topk_ids=ids,
        reference_topk_logp=ref_top,
        reference_residual_log_mass=ref_residual,
    )
    torch.testing.assert_close(js, torch.zeros_like(js), atol=1e-7, rtol=0)
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-7, rtol=0)


def test_residual_combined_requires_benefit_and_unlearned_gain() -> None:
    scores = residual_combined_scores(
        teacher_target_logp=np.array([-1.0, -1.0, -2.0]),
        student_target_logp=np.array([-2.0, -0.5, -3.0]),
        teacher_beneficial=np.array([True, True, False]),
        teacher_student_js=np.array([0.2, 0.3, 0.4]),
    )
    np.testing.assert_allclose(scores, [0.2, 0.0, 0.0])


def test_selection_and_anchor_are_disjoint_and_deterministic() -> None:
    selected = select_positive_top_fraction(np.array([1.0, 4.0, 3.0, 0.0]), 0.5)
    assert selected == [1, 2]
    anchor = select_harmful_anchor_positions(
        np.array([0.3, 9.0, 8.0, 0.7]),
        count=2,
        excluded=set(selected),
        teacher_beneficial=np.array([False, True, True, False]),
    )
    assert anchor == [0, 3]


def test_pair_mask_and_shrink_leave_six_rows_bit_identical() -> None:
    prefix = torch.arange(24, dtype=torch.float32).view(8, 3)
    start = prefix.clone()
    gradient = torch.ones_like(prefix)
    gradient[5] *= 20
    gradient[2] *= 10
    pair, _ = choose_prefix_pair(gradient, prefix + 1.0)
    assert pair == [2, 5]
    mask_prefix_gradient_(gradient, pair)
    for index in range(8):
        if index not in pair:
            assert torch.count_nonzero(gradient[index]) == 0
    prefix[2] += 4
    prefix[5] -= 8
    prefix[0] += 100  # Simulate an accidental frozen-row drift.
    shrink_pair_update_(
        torch,
        prefix,
        round_start=start,
        active_rows=pair,
        shrinkage=0.25,
    )
    torch.testing.assert_close(prefix[2], start[2] + 1)
    torch.testing.assert_close(prefix[5], start[5] - 2)
    for index in range(8):
        if index not in pair:
            assert torch.equal(prefix[index], start[index])
