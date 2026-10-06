from canirunlocalai.benchmark import _ollama_sample, _rate, _require_loopback, summarize_samples


def test_rate():
    assert _rate(100, 2_000_000_000) == 50.0


def test_ollama_sample_extracts_server_metrics():
    row = _ollama_sample(10.0, 10.25, 11.0, {
        "eval_count": 50, "eval_duration": 1_000_000_000,
        "prompt_eval_count": 100, "prompt_eval_duration": 500_000_000,
        "load_duration": 100_000_000, "total_duration": 1_100_000_000,
    }, "hello")
    assert row["ttft_ms"] == 250.0
    assert row["decode_tps"] == 50.0
    assert row["prompt_tps"] == 200.0
    assert row["load_ms"] == 100.0


def test_summary_uses_medians_and_ignores_missing():
    summary = summarize_samples([
        {"ttft_ms": 100, "wall_ms": 500, "load_ms": None, "prompt_tps": 100, "decode_tps": 20},
        {"ttft_ms": 200, "wall_ms": 700, "load_ms": 10, "prompt_tps": 200, "decode_tps": 40},
        {"ttft_ms": 300, "wall_ms": 900, "load_ms": 20, "prompt_tps": None, "decode_tps": 60},
    ])
    assert summary["median_ttft_ms"] == 200.0
    assert summary["median_decode_tps"] == 40.0
    assert summary["median_load_ms"] == 15.0


def test_remote_endpoint_is_rejected():
    try:
        _require_loopback("https://example.com")
    except ValueError:
        return
    raise AssertionError("remote endpoint should be rejected")
