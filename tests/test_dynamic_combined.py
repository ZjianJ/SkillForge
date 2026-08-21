from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import scripts.train_spreadsheetbench_dynamic_combined as dynamic_trainer
from skillopt.softprefix.dynamic_combined import (
    dynamic_additive_scores,
    dynamic_gain_competitor_scores,
    dynamic_skill_effect_scores,
    dynamic_stop_decision,
    full_vocab_dynamic_metrics,
    jaccard_indices,
    locator_loss_weights,
    select_dynamic_top_fraction,
)


def test_dynamic_additive_scores_match_per_trajectory_normalization() -> None:
    result = dynamic_additive_scores(
        np.array([0.0, 2.0, 4.0, 100.0]),
        np.array([3.0, 2.0, 1.0, 100.0]),
        alpha=0.5,
        eligible_indices=[0, 1, 2],
    )
    np.testing.assert_allclose(result["normalized_gain"], [0.0, 0.5, 1.0, 0.0])
    np.testing.assert_allclose(result["normalized_js"], [1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(result["additive"], [0.5, 0.5, 0.5, 0.0])


def test_dynamic_additive_scores_reject_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        dynamic_additive_scores(np.ones(2), np.ones(2), alpha=1.1)


def test_dynamic_gain_competitor_scores_match_normalization() -> None:
    result = dynamic_gain_competitor_scores(
        np.array([0.0, 2.0, 4.0, 100.0]),
        np.array([4.0, 2.0, 0.0, 100.0]),
        gain_weight=0.75,
        eligible_indices=[0, 1, 2],
    )
    np.testing.assert_allclose(result["normalized_gain"], [0.0, 0.5, 1.0, 0.0])
    np.testing.assert_allclose(
        result["normalized_competitor"], [1.0, 0.5, 0.0, 0.0]
    )
    np.testing.assert_allclose(result["gain_competitor"], [0.25, 0.5, 0.75, 0.0])


def test_locator_loss_weights_preserve_requested_effective_mass() -> None:
    weights = locator_loss_weights(
        np.array([0.1, 0.4, 0.8, 0.2]),
        [1, 2, 3],
        effective_weight=2.0,
    )
    assert weights.sum() == pytest.approx(2.0)
    np.testing.assert_allclose(weights / weights[0], [1.0, 2.0, 0.5])


def test_full_vocab_dynamic_metrics_matches_naive() -> None:
    teacher = torch.tensor(
        [[2.0, 0.0, -1.0, 0.5], [0.0, 1.0, 2.0, -0.5]], dtype=torch.float32
    )
    student = torch.tensor(
        [[0.5, 0.0, 1.0, -1.0], [0.0, 2.0, 0.5, -0.5]], dtype=torch.float32
    )
    target = torch.tensor([0, 2])
    result = full_vocab_dynamic_metrics(
        teacher_logits=teacher,
        student_logits=student,
        target_ids=target,
        original_beneficial=np.array([True, False]),
        chunk_size=1,
    )

    tq = torch.log_softmax(teacher, dim=-1)
    sp = torch.log_softmax(student, dim=-1)
    q, p = tq.exp(), sp.exp()
    mix = 0.5 * (q + p)
    expected_kl = (q * (tq - sp)).sum(dim=-1).numpy()
    expected_js = 0.5 * (
        (q * (tq - mix.log())).sum(dim=-1)
        + (p * (sp - mix.log())).sum(dim=-1)
    ).numpy()
    expected_gain = torch.clamp(tq.gather(1, target[:, None]).squeeze(1) - sp.gather(1, target[:, None]).squeeze(1), min=0).numpy()
    teacher_wrong = torch.stack([tq[0, 3], tq[1, 1]])
    student_wrong = torch.stack([sp[0, 2], sp[1, 1]])
    expected_competitor = torch.clamp(student_wrong - teacher_wrong, min=0).numpy()

    np.testing.assert_allclose(result["forward_kl"], expected_kl, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result["js"], expected_js, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result["residual_gain"], expected_gain, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        result["competitor_suppression"], expected_competitor, rtol=1e-6, atol=1e-6
    )
    expected_teacher_entropy = -(q * tq).sum(dim=-1).numpy()
    expected_student_entropy = -(p * sp).sum(dim=-1).numpy()
    expected_resolved = np.maximum(expected_student_entropy - expected_teacher_entropy, 0.0) * (
        1.0 - expected_teacher_entropy / math.log(teacher.shape[-1])
    )
    np.testing.assert_allclose(result["teacher_entropy"], expected_teacher_entropy, rtol=1e-6)
    np.testing.assert_allclose(result["student_entropy"], expected_student_entropy, rtol=1e-6)
    np.testing.assert_allclose(result["resolved_uncertainty"], expected_resolved, rtol=1e-6)
    assert result["combined"][0] == pytest.approx(expected_gain[0] * expected_js[0])
    assert result["combined"][1] == 0.0


def test_four_signal_additive_and_multiplicative_scores() -> None:
    signals = (
        np.array([0.0, 1.0, 2.0]),
        np.array([2.0, 1.0, 0.0]),
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 1.0, 2.0]),
    )
    additive = dynamic_skill_effect_scores(
        *signals, weights=[0.4, 0.4, 0.1, 0.1], mode="additive"
    )
    np.testing.assert_allclose(additive["skill_effect"], [0.4, 0.5, 0.6])
    multiplicative = dynamic_skill_effect_scores(
        *signals, weights=[0.45, 0.45, 0.10, 0.25], mode="multiplicative"
    )
    np.testing.assert_allclose(multiplicative["skill_effect"], [0.45, 0.5625, 0.6875])


def test_four_signal_scores_validate_weight_contract() -> None:
    values = np.ones(2)
    with pytest.raises(ValueError):
        dynamic_skill_effect_scores(
            values, values, values, values, weights=[0.4, 0.4, 0.1, 0.2]
        )
    with pytest.raises(ValueError):
        dynamic_skill_effect_scores(
            values,
            values,
            values,
            values,
            weights=[0.45, 0.45, 0.2, 0.5],
            mode="multiplicative",
        )


def test_meta_weight_change_is_normalized_and_bounded() -> None:
    old = np.array([0.4, 0.3, 0.2, 0.1])
    updated = dynamic_trainer._cap_weight_change(
        np.array([0.8, 0.1, 0.05, 0.05]), old, 0.1
    )
    assert updated.sum() == pytest.approx(1.0)
    assert np.max(np.abs(updated - old)) <= 0.1 + 1e-6


def test_dynamic_selection_is_per_trajectory_stable_and_excludes_forbidden() -> None:
    scores = np.array([0.1, 0.9, 0.9, 0.8, 0.0, np.nan], dtype=np.float32)
    # ceil(6 * .5) = 3; index 1 is forbidden and equal scores retain index order.
    assert select_dynamic_top_fraction(scores, ratio=0.5, forbidden_indices=[1]) == [0, 2, 3]
    assert select_dynamic_top_fraction(np.zeros(4), ratio=0.5) == []
    with pytest.raises(ValueError):
        select_dynamic_top_fraction(scores, ratio=1.0)


def test_jaccard_indices_handles_empty_sets() -> None:
    assert jaccard_indices([], []) == 1.0
    assert jaccard_indices([1, 2], [2, 3]) == pytest.approx(1 / 3)


def test_dynamic_stop_small_residual_and_budget() -> None:
    small = dynamic_stop_decision(
        current_mass=9.0,
        initial_mass=100.0,
        previous_mass=20.0,
        mean_eligible_kl=0.2,
        completed_relocations=2,
        max_relocations=4,
        stagnant_rounds=0,
        residual_mass_ratio_threshold=0.1,
        mean_eligible_kl_threshold=0.01,
        min_relative_mass_improvement=0.002,
        global_patience=2,
    )
    assert small.stop and small.reason == "residual-mass-small"
    assert small.residual_mass_ratio == pytest.approx(0.09)

    budget = dynamic_stop_decision(
        current_mass=50.0,
        initial_mass=100.0,
        previous_mass=60.0,
        mean_eligible_kl=0.2,
        completed_relocations=4,
        max_relocations=4,
        stagnant_rounds=0,
        residual_mass_ratio_threshold=0.1,
        mean_eligible_kl_threshold=0.01,
        min_relative_mass_improvement=0.002,
        global_patience=2,
    )
    assert budget.stop and budget.reason == "relocation-budget-exhausted"


def test_dynamic_stop_tracks_global_stagnation() -> None:
    first = dynamic_stop_decision(
        current_mass=99.9,
        initial_mass=100.0,
        previous_mass=100.0,
        mean_eligible_kl=0.2,
        completed_relocations=1,
        max_relocations=4,
        stagnant_rounds=0,
        residual_mass_ratio_threshold=0.1,
        mean_eligible_kl_threshold=0.01,
        min_relative_mass_improvement=0.002,
        global_patience=2,
    )
    assert not first.stop and first.stagnant_rounds == 1
    assert first.relative_mass_improvement == pytest.approx(0.001)

    second = dynamic_stop_decision(
        current_mass=99.8,
        initial_mass=100.0,
        previous_mass=99.9,
        mean_eligible_kl=0.2,
        completed_relocations=2,
        max_relocations=4,
        stagnant_rounds=first.stagnant_rounds,
        residual_mass_ratio_threshold=0.1,
        mean_eligible_kl_threshold=0.01,
        min_relative_mass_improvement=0.002,
        global_patience=2,
    )
    assert second.stop and second.reason == "global-residual-stagnant"
    assert second.stagnant_rounds == 2
    assert math.isfinite(second.residual_mass_ratio)


def test_dynamic_localize_writes_auditable_core_and_excludes_preserve(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Example:
        task_id = "task/one"
        target_ids = [0, 1, 2, 3, 4]

    class PrefixModel:
        torch = torch

    teacher = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, 0.0],
        ]
    )
    student = torch.zeros_like(teacher)

    def fake_target_logits(prefix_model, example, indices, *, use_prefix, with_grad):
        del prefix_model, example, with_grad
        source = student if use_prefix else teacher
        return source[torch.as_tensor(indices, dtype=torch.long)]

    monkeypatch.setattr(dynamic_trainer, "target_logits", fake_target_logits)
    record = {
        "example": Example(),
        "selected": [],
        "preserve": [0],
        "original_beneficial": np.ones(4, dtype=bool),
        "clean_topk_ids": np.array([[0, 1]], dtype=np.int64),
        "clean_topk_logp": np.log(np.array([[0.3, 0.2]], dtype=np.float32)),
        "clean_residual": np.log(np.array([0.5], dtype=np.float32)),
    }
    summary, selected = dynamic_trainer._dynamic_localize(
        PrefixModel(),
        [record],
        round_index=0,
        out_root=tmp_path,
        dynamic_cfg={
            "core_ratio": 0.5,
            "kl_chunk_size": 2,
            "exclude_preservation_from_core": True,
        },
        previous=None,
    )
    assert 0 not in selected["task/one"]
    assert len(selected["task/one"]) == 2
    assert summary["selected_tokens"] == 2
    assert summary["selected_mass_capture"] > 0
    assert (tmp_path / "locators/round_00/summary.json").is_file()
    assert (tmp_path / "locators/round_00/manifest.jsonl").is_file()
    assert (tmp_path / "locators/round_00/arrays/task_one.npz").is_file()


def test_dynamic_additive_localize_uses_fixed_budget_without_positive_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Example:
        task_id = "additive/task"
        target_ids = [0, 1, 2, 3, 4]

    class PrefixModel:
        torch = torch

    teacher = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, 0.0],
        ]
    )
    student = torch.zeros_like(teacher)

    def fake_target_logits(prefix_model, example, indices, *, use_prefix, with_grad):
        del prefix_model, example, with_grad
        source = student if use_prefix else teacher
        return source[torch.as_tensor(indices, dtype=torch.long)]

    monkeypatch.setattr(dynamic_trainer, "target_logits", fake_target_logits)
    record = {
        "example": Example(),
        "selected": [],
        "preserve": [0],
        # Additive Skill must not be restricted by the legacy beneficial gate.
        "original_beneficial": np.zeros(4, dtype=bool),
        "clean_topk_ids": np.array([[0, 1]], dtype=np.int64),
        "clean_topk_logp": np.log(np.array([[0.3, 0.2]], dtype=np.float32)),
        "clean_residual": np.log(np.array([0.5], dtype=np.float32)),
    }
    summary, selected = dynamic_trainer._dynamic_localize(
        PrefixModel(),
        [record],
        round_index=0,
        out_root=tmp_path,
        dynamic_cfg={
            "core_ratio": 0.5,
            "kl_chunk_size": 2,
            "exclude_preservation_from_core": True,
            "locator_method": "additive_skill",
            "additive_alpha": 0.5,
        },
        previous=None,
    )
    assert 0 not in selected["additive/task"]
    assert len(selected["additive/task"]) == 2
    assert summary["locator_method"] == "additive_skill"
    assert summary["global_full_vocab_kl_mass"] > 0


def test_dynamic_additive_localize_weights_top20_to_top10_effective_mass(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Example:
        task_id = "weighted/task"
        target_ids = list(range(11))

    class PrefixModel:
        torch = torch

    teacher = torch.eye(10, 11) * 3.0
    student = torch.zeros_like(teacher)

    def fake_target_logits(prefix_model, example, indices, *, use_prefix, with_grad):
        del prefix_model, example, with_grad
        source = student if use_prefix else teacher
        return source[torch.as_tensor(indices, dtype=torch.long)]

    monkeypatch.setattr(dynamic_trainer, "target_logits", fake_target_logits)
    record = {
        "example": Example(),
        "selected": [],
        "preserve": [],
        "original_beneficial": np.ones(10, dtype=bool),
        "clean_topk_ids": np.empty((0, 2), dtype=np.int64),
        "clean_topk_logp": np.empty((0, 2), dtype=np.float32),
        "clean_residual": np.empty(0, dtype=np.float32),
    }
    summary, _ = dynamic_trainer._dynamic_localize(
        PrefixModel(),
        [record],
        round_index=0,
        out_root=tmp_path,
        dynamic_cfg={
            "core_ratio": 0.20,
            "effective_core_ratio": 0.10,
            "loss_weighting": "locator_score",
            "locator_method": "additive_skill",
        },
        previous=None,
    )
    assert len(record["selected"]) == 2
    assert record["effective_skill_weight"] == pytest.approx(1.0)
    assert summary["selected_effective_weight"] == pytest.approx(1.0)
