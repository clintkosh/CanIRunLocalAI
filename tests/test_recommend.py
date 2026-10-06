from canirunlocalai.recommend import classify_device, recommend_models, score_model


def fake_hardware(vram_gb=8, ram_gb=32):
    return {
        "memory": {"total_gb": ram_gb, "available_gb": ram_gb * 0.65},
        "gpus": [{"name": "Test GPU", "vendor": "NVIDIA", "vram_total_gb": vram_gb, "vram_free_gb": vram_gb * 0.85}],
        "runtimes": {"ollama": {"installed": True}, "llama_cpp": {"installed": False}},
    }


def test_classify_device_entry_gpu():
    assert classify_device(fake_hardware(8, 32)) == "entry-to-midrange GPU system"


def test_recommends_models_for_8gb_gpu():
    result = recommend_models(fake_hardware(8, 32), use_case="chat", limit=5)
    names = [row["model"] for row in result["recommendations"]]
    assert any(name in names for name in ["qwen3.5:4b", "qwen3:8b", "llama3.2:3b"])


def test_latency_objective_prefers_compact_current_models():
    result = recommend_models(fake_hardware(6, 64), use_case="chat", objective="latency", limit=5)
    names = [row["model"] for row in result["recommendations"]]
    assert "qwen3.5:4b" in names
    if "qwen3.5:9b" in names:
        assert names.index("qwen3.5:4b") < names.index("qwen3.5:9b")


def test_too_large_model_not_great_fit():
    model = {
        "id": "huge:test", "family": "Huge", "estimated_q4_gb": 70,
        "min_ram_gb": 128, "min_vram_gb": 80, "params_b": 120,
        "use_cases": ["chat"], "quality_tier": 10,
    }
    row = score_model(model, fake_hardware(8, 32), use_case="chat")
    assert row["fit"] == "not_recommended"
