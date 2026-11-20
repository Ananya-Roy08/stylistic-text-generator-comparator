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
def test_milestone_4(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import torch
    from torch import nn
    from torch.nn import functional as F

    import app.core as core

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.projections = nn.Module()
            self.projections.c_attn = nn.Linear(4, 4)
            self.projections.c_proj = nn.Linear(4, 4)
            self.output = nn.Linear(4, 8)
            core.inject_lora_adapters(self, rank=2)

        def forward(self, input_ids, attention_mask, labels):
            hidden = self.embedding(input_ids)
            hidden = self.projections.c_attn(hidden)
            hidden = self.projections.c_proj(hidden)
            logits = self.output(hidden)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            return SimpleNamespace(loss=loss)

    class TinyTokenizer:
        pad_token_id = 0
        eos_token = "</s>"
        pad_token = None

        def __call__(self, texts, **kwargs):
            input_ids = torch.tensor([[1, 3, 4, 2] for _ in texts])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

    model = TinyModel()
    monkeypatch.setattr(core, "milestone_3", lambda: (model, TinyTokenizer()))
    monkeypatch.setattr(core, "ADAPTER_WEIGHTS_PATH", tmp_path / "adapter.pt")

    initial_b = model.projections.c_attn.lora_B.weight.detach().clone()
    adapter_weights = core.milestone_4()
    saved_weights = torch.load(tmp_path / "adapter.pt", weights_only=True)

    assert adapter_weights
    assert any("lora_" in name for name in adapter_weights)
    assert any(
        not torch.equal(initial_b, tensor)
        for name, tensor in adapter_weights.items()
        if "c_attn.lora_B.weight" in name
    )
    assert set(saved_weights) == set(adapter_weights)
    assert all(torch.equal(saved_weights[name], value) for name, value in adapter_weights.items())


# Milestone 5: Generate text from the LoRA-adapted LLM.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

