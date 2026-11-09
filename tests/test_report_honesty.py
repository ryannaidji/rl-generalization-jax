"""The report must say Part B has no results for as long as no result file exists."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULT = ROOT / "results/part_b/dpo_results.json"


def test_report_flags_part_b_not_run_until_results_exist():
    if not RESULT.exists():
        assert "Part B has no results until" in (ROOT / "report/report.md").read_text(encoding="utf-8")


def test_part_b_result_is_a_real_gpu_run():
    if RESULT.exists():
        r = json.loads(RESULT.read_text())
        assert r["smoke_run_not_a_result"] is False and r["hardware"]["gpu"]


def test_notebook_is_valid():
    nb = json.loads((ROOT / "colab/dpo_colab.ipynb").read_text(encoding="utf-8"))
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    assert any("dpo_loss" in "".join(c["source"]) for c in nb["cells"])
