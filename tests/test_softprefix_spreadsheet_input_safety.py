from pathlib import Path

from skillopt.softprefix import trainer


def test_spreadsheet_eval_executes_against_copy(monkeypatch, tmp_path: Path) -> None:
    canonical_input = tmp_path / "canonical.xlsx"
    canonical_input.write_bytes(b"original workbook bytes")
    prediction = tmp_path / "prediction.xlsx"
    observed: dict[str, Path] = {}

    def destructive_executor(code: str, input_path: str, output_path: str, timeout: int):
        del code, timeout
        exposed_input = Path(input_path)
        observed["input"] = exposed_input
        Path(output_path).write_bytes(exposed_input.read_bytes())
        exposed_input.unlink()
        return True, ""

    monkeypatch.setattr(trainer, "run_spreadsheet_generated_code", destructive_executor)

    ok, error = trainer._run_spreadsheet_generated_code_on_copy(
        "destructive code", str(canonical_input), str(prediction), timeout=10
    )

    assert ok, error
    assert observed["input"] != canonical_input
    assert canonical_input.read_bytes() == b"original workbook bytes"
    assert prediction.read_bytes() == b"original workbook bytes"
