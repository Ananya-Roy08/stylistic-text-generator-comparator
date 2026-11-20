"""Behavior tests for the comparator's models and workflows."""
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
def test_milestone_5(monkeypatch, tmp_path):
    import torch
    from torch import nn

    import app.core as core

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.projections = nn.Module()
            self.projections.c_attn = nn.Linear(4, 4)
            self.projections.c_proj = nn.Linear(4, 4)
            core.inject_lora_adapters(self, rank=2)
            self.generation_args = None

        def generate(self, input_ids, attention_mask, **kwargs):
            self.generation_args = kwargs
            return torch.cat((input_ids, torch.tensor([[5, 6]])), dim=1)

    class TinyTokenizer:
        eos_token_id = 2
        pad_token_id = 0
        pad_token = None
        eos_token = "</s>"

        def __call__(self, text, return_tensors):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

        def decode(self, tokens, skip_special_tokens):
            return "The opening delivery swings away."

    model = TinyModel()
    tokenizer = TinyTokenizer()
    monkeypatch.setattr(core, "milestone_3", lambda: (model, tokenizer))
    monkeypatch.setattr(core, "ADAPTER_WEIGHTS_PATH", tmp_path / "adapter.pt")
    adapter_weights = {
        name: torch.full_like(tensor, 0.125)
        for name, tensor in model.state_dict().items()
        if "lora_" in name
    }
    torch.save(adapter_weights, core.ADAPTER_WEIGHTS_PATH)

    generated = core.milestone_5(
        "The opening delivery",
        max_new_tokens=12,
        temperature=0.7,
        top_p=0.85,
    )

    assert generated == "The opening delivery swings away."
    assert model.generation_args == {
        "max_new_tokens": 12,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.85,
        "pad_token_id": 0,
    }
    assert not model.training
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in adapter_weights.items()
    )
    with pytest.raises(ValueError, match="top_p"):
        core.milestone_5("seed", top_p=0)


def test_milestone_5_requires_saved_adapter_weights(monkeypatch, tmp_path):
    import app.core as core

    monkeypatch.setattr(core, "ADAPTER_WEIGHTS_PATH", tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="milestone_4"):
        core.milestone_5("seed")


# Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
def test_milestone_6(monkeypatch):
    import app.core as core

    calls = []

    class TinyNGram:
        def generate(self, seed_prompt, max_new_tokens):
            calls.append(("ngram", seed_prompt, max_new_tokens))
            return f"{seed_prompt} from ngram"

    def generate_lora(seed_prompt, max_new_tokens, temperature, top_p):
        calls.append(("lora", seed_prompt, max_new_tokens, temperature, top_p))
        return f"{seed_prompt} from lora"

    monkeypatch.setattr(core, "milestone_2", lambda: TinyNGram())
    monkeypatch.setattr(core, "milestone_5", generate_lora)

    outputs = core.milestone_6(
        "The opening delivery",
        max_new_tokens=12,
        temperature=0.7,
        top_p=0.85,
    )

    assert outputs == {
        "ngram": "The opening delivery from ngram",
        "lora": "The opening delivery from lora",
    }
    assert calls == [
        ("ngram", "The opening delivery", 12),
        ("lora", "The opening delivery", 12, 0.7, 0.85),
    ]

