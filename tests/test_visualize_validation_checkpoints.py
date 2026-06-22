from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "visualize_validation_checkpoints.py"
    spec = importlib.util.spec_from_file_location("visualize_validation_checkpoints", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_summary(path: Path, summary: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f)


def test_collect_records_uses_last_accepted_checkpoint(tmp_path: Path) -> None:
    module = load_module()
    run_dir = tmp_path / "outputs" / "skill_section" / "main_searchqa_seed1"
    write_summary(
        run_dir,
        {
            "best_score": 0.8,
            "history": [
                {"epoch": 1, "loss": 1.2, "valid_seen_score": 0.7, "valid_seen_soft": 0.7, "action": "accept_new_best"},
                {"epoch": 2, "loss": 1.0, "valid_seen_score": 0.8, "valid_seen_soft": 0.8, "action": "accept_new_best"},
                {"epoch": 3, "loss": 1.4, "valid_seen_score": 0.6, "valid_seen_soft": 0.6, "action": "reject"},
            ],
        },
    )

    records = module.collect_records(tmp_path / "outputs", metric="valid_seen_score")

    assert len(records) == 1
    assert records[0].run == "main_searchqa_seed1"
    assert records[0].setting == "skill_section"
    assert records[0].task == "searchqa"
    assert records[0].seed == "1"
    assert records[0].best_epoch == 2
    assert records[0].epoch_scores == {1: 0.7, 2: 0.8, 3: 0.6}
    assert records[0].epoch_losses == {1: 1.2, 2: 1.0, 3: 1.4}


def test_collect_records_falls_back_to_first_max_score_for_ties(tmp_path: Path) -> None:
    module = load_module()
    run_dir = tmp_path / "outputs" / "prompt_start" / "main_docvqa_seed2"
    write_summary(
        run_dir,
        {
            "best_score": 0.9,
            "history": [
                {"epoch": 1, "valid_seen_score": 0.9, "valid_seen_soft": 0.9},
                {"epoch": 2, "valid_seen_score": 0.8, "valid_seen_soft": 0.8},
                {"epoch": 3, "valid_seen_score": 0.9, "valid_seen_soft": 0.9},
            ],
        },
    )

    records = module.collect_records(tmp_path / "outputs", metric="valid_seen_score")

    assert len(records) == 1
    assert records[0].best_epoch == 1


def test_summarize_best_epochs_counts_frequency_by_epoch() -> None:
    module = load_module()
    records = [
        module.ValidationRecord(
            summary_path=Path("a/summary.json"),
            run_dir=Path("a"),
            setting="skill_section",
            run="main_a_seed1",
            task="a",
            seed="1",
            family="main",
            best_epoch=1,
            best_score=0.5,
            epoch_scores={1: 0.5, 2: 0.4, 3: 0.3},
            epoch_losses={1: 1.2, 2: 1.3, 3: 1.5},
        ),
        module.ValidationRecord(
            summary_path=Path("b/summary.json"),
            run_dir=Path("b"),
            setting="skill_section",
            run="main_b_seed1",
            task="b",
            seed="1",
            family="main",
            best_epoch=3,
            best_score=0.7,
            epoch_scores={1: 0.5, 2: 0.6, 3: 0.7},
            epoch_losses={1: 1.3, 2: 1.1, 3: 0.9},
        ),
    ]

    counts = module.summarize_best_epochs(records)

    assert counts == [
        {"epoch": 1, "count": 1, "frequency": 0.5},
        {"epoch": 2, "count": 0, "frequency": 0.0},
        {"epoch": 3, "count": 1, "frequency": 0.5},
    ]


def test_group_records_by_task_splits_benchmarks() -> None:
    module = load_module()
    searchqa = module.ValidationRecord(
        summary_path=Path("searchqa/summary.json"),
        run_dir=Path("searchqa"),
        setting="skill_section",
        run="main_searchqa_seed1",
        task="searchqa",
        seed="1",
        family="main",
        best_epoch=2,
        best_score=0.8,
        epoch_scores={1: 0.7, 2: 0.8, 3: 0.6},
        epoch_losses={1: 1.2, 2: 1.0, 3: 1.4},
    )
    docvqa = module.ValidationRecord(
        summary_path=Path("docvqa/summary.json"),
        run_dir=Path("docvqa"),
        setting="skill_section",
        run="main_docvqa_seed1",
        task="docvqa",
        seed="1",
        family="main",
        best_epoch=1,
        best_score=0.5,
        epoch_scores={1: 0.5, 2: 0.4, 3: 0.45},
        epoch_losses={1: 0.9, 2: 1.1, 3: 1.0},
    )

    grouped = module.group_records_by_task([searchqa, docvqa])

    assert grouped == {"docvqa": [docvqa], "searchqa": [searchqa]}


def test_build_heatmap_matrix_uses_delta_from_best_score() -> None:
    module = load_module()
    record = module.ValidationRecord(
        summary_path=Path("searchqa/summary.json"),
        run_dir=Path("searchqa"),
        setting="skill_section",
        run="main_searchqa_seed1",
        task="searchqa",
        seed="1",
        family="main",
        best_epoch=2,
        best_score=0.8,
        epoch_scores={1: 0.7, 2: 0.8, 3: 0.6},
        epoch_losses={1: 1.2, 2: 1.0, 3: 1.4},
    )

    epochs, rows, matrix = module.build_heatmap_matrix([record])

    assert epochs == [1, 2, 3]
    assert rows == [record]
    assert matrix == [[-0.10000000000000009, 0.0, -0.20000000000000007]]


def test_build_loss_accuracy_points_pairs_epochs_with_both_metrics() -> None:
    module = load_module()
    record = module.ValidationRecord(
        summary_path=Path("searchqa/summary.json"),
        run_dir=Path("searchqa"),
        setting="skill_section",
        run="main_searchqa_seed1",
        task="searchqa",
        seed="1",
        family="main",
        best_epoch=2,
        best_score=0.8,
        epoch_scores={1: 0.7, 2: 0.8, 3: 0.6},
        epoch_losses={1: 1.2, 2: 1.0},
    )

    points = module.build_loss_accuracy_points([record])

    assert points == [
        (record, 1, 1.2, 0.7),
        (record, 2, 1.0, 0.8),
    ]


def test_write_artifacts_creates_svg_without_png_backend(tmp_path: Path) -> None:
    module = load_module()
    record = module.ValidationRecord(
        summary_path=Path("searchqa/summary.json"),
        run_dir=Path("searchqa"),
        setting="skill_section",
        run="main_searchqa_seed1",
        task="searchqa",
        seed="1",
        family="main",
        best_epoch=2,
        best_score=0.8,
        epoch_scores={1: 0.7, 2: 0.8, 3: 0.6},
        epoch_losses={1: 1.2, 2: 1.0, 3: 1.4},
    )

    module.write_artifacts([record], tmp_path, metric="valid_seen_score", dpi=90, plot_png=False)

    assert (tmp_path / "best_epoch_frequency.svg").is_file()
    assert (tmp_path / "validation_accuracy_heatmap.svg").is_file()
    assert (tmp_path / "loss_vs_validation_accuracy.svg").is_file()
