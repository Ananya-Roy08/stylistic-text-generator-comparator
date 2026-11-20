"""Smoke tests for storage, configuration, and CLI entry points."""

from app import llm
from app.storage import Store


def test_the_store_keeps_what_you_put_in_it(tmp_path):
    store = Store(str(tmp_path / "app.json"))
    store.put("hello", {"n": 1})
    assert store.get("hello") == {"n": 1}
    assert store.get("missing", "fallback") == "fallback"
    assert store.all() == {"hello": {"n": 1}}


def test_model_settings_default_to_a_local_ollama(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                  # no .env here
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    base, key, model = llm.settings()
    assert base == "http://localhost:11434/v1" and key == "" and model == ""


def test_the_command_line_runs(capsys):
    from app.cli import main
    main(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_compare_command_displays_both_outputs(monkeypatch, capsys):
    from app import core
    from app.cli import main

    def compare(seed_prompt, max_new_tokens, temperature, top_p):
        assert seed_prompt == "The opening delivery"
        assert (max_new_tokens, temperature, top_p) == (12, 0.7, 0.85)
        return {
            "ngram": "The opening delivery from the N-gram model.",
            "lora": "The opening delivery from the LoRA model.",
        }

    monkeypatch.setattr(core, "milestone_6", compare)
    main([
        "compare",
        "The opening delivery",
        "--max-new-tokens",
        "12",
        "--temperature",
        "0.7",
        "--top-p",
        "0.85",
    ])
    output = capsys.readouterr().out
    assert "Seed prompt: The opening delivery" in output
    assert "N-gram" in output and "LoRA-adapted LLM" in output
    assert "from the N-gram model." in output
    assert "from the LoRA model." in output

