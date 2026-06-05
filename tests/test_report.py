from canirunlocalai.report import make_markdown_report
from canirunlocalai.recommend import recommend_models


def test_markdown_report_contains_sections():
    hardware = {
        "system": {"os": "TestOS", "os_release": "1", "machine": "x86_64"},
        "cpu": {"brand": "Test CPU", "physical_cores": 4, "logical_cores": 8},
        "memory": {"total_gb": 16, "available_gb": 10},
        "storage": [],
        "gpus": [],
        "runtimes": {"ollama": {"installed": False, "server_running": False}, "docker": {"installed": False}},
        "notes": [],
    }
    recs = recommend_models(hardware, limit=3)
    md = make_markdown_report(hardware, recs)
    assert "# CanIRunLocalAI Report" in md
    assert "## Recommended local models" in md
