"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Prepare a small corpus of sports commentary text.
def test_milestone_1():
    from app.core import milestone_1

    commentary = milestone_1()
    assert len(commentary) >= 12
    assert all(line == line.strip() and line for line in commentary)
    assert any("wicket" in line.lower() for line in commentary)


# Milestone 2: Implement the n-gram model and generate text from it.
def test_milestone_2():
    from app.core import milestone_2

    model = milestone_2()
    generated = model.generate("The opening delivery", max_new_tokens=20)
    assert generated.startswith("The opening delivery ")
    assert generated.endswith(".")
    assert len(generated.split()) <= len("The opening delivery".split()) + 20
    assert model.generate("The opening delivery", max_new_tokens=0) == "The opening delivery"

    with pytest.raises(ValueError, match="seed_prompt"):
        model.generate("   ")


# Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
def test_milestone_3(monkeypatch):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

    from app.core import LoRALinear, milestone_3

    model = AutoModelForCausalLM.from_config(
        GPT2Config(
            n_layer=1,
            n_head=2,
            n_embd=8,
            vocab_size=32,
            n_positions=16,
            n_ctx=16,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3]])
    expected_logits = model(input_ids=input_ids).logits.detach()
    tokenizer = object()
    calls = []

    def load_model(cls, model_name):
        calls.append(("model", model_name))
        return model

    def load_tokenizer(cls, model_name):
        calls.append(("tokenizer", model_name))
        return tokenizer

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(load_model))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(load_tokenizer))

    loaded_model, loaded_tokenizer = milestone_3()
    assert calls == [("tokenizer", "distilgpt2"), ("model", "distilgpt2")]
    assert loaded_tokenizer is tokenizer
    assert isinstance(loaded_model.transformer.h[0].attn.c_attn, LoRALinear)
    assert isinstance(loaded_model.transformer.h[0].attn.c_proj, LoRALinear)
    assert all(
        not parameter.requires_grad
        for name, parameter in loaded_model.named_parameters()
        if "lora_" not in name
    )
    assert any(parameter.requires_grad for parameter in loaded_model.parameters())
    assert torch.allclose(loaded_model(input_ids=input_ids).logits, expected_logits)


# Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Generate text from the LoRA-adapted LLM.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

