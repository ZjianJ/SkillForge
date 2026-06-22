from __future__ import annotations

import openpyxl

from skillopt.envs.spreadsheetbench.rollout import _auto_verify_output


def _write_workbook(path, value: object = 1) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    worksheet["A1"] = value
    workbook.save(path)
    workbook.close()


def test_auto_verify_reports_invalid_answer_position_without_raising(tmp_path) -> None:
    pred_path = tmp_path / "pred.xlsx"
    gold_path = tmp_path / "gold.xlsx"
    _write_workbook(pred_path)
    _write_workbook(gold_path)

    report = _auto_verify_output(
        str(pred_path),
        str(gold_path),
        "Consolidated Tracker",
    )

    assert "Invalid answer position" in report
    assert "Consolidated Tracker" in report


def test_auto_verify_reports_non_cell_answer_position_without_raising(tmp_path) -> None:
    pred_path = tmp_path / "pred.xlsx"
    gold_path = tmp_path / "gold.xlsx"
    _write_workbook(pred_path)
    _write_workbook(gold_path)

    report = _auto_verify_output(
        str(pred_path),
        str(gold_path),
        "A",
    )

    assert "Invalid answer position" in report
    assert "A" in report
