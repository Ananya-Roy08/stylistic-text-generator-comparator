"""Your code. One stub per milestone; build them in order and keep each one small."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


ADAPTER_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "data" / "lora_adapter.pt"


# --- Milestone 1: Prepare a small corpus of sports commentary text.
# Hint: You'll need to gather and clean a dataset suitable for training.
def milestone_1() -> list[str]:
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "sports_commentary.txt"
    return [
        line.strip()
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class NGramModel:
    def __init__(self, corpus: list[str], n: int = 3) -> None:
        if n < 2:
            raise ValueError("n must be at least 2")

        self.n = n
        self._continuations: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for line in corpus:
            tokens = line.split()
            for index, token in enumerate(tokens):
                for context_size in range(min(n - 1, index) + 1):
                    context = tuple(tokens[index - context_size : index])
                    self._continuations[context][token] += 1

        if not self._continuations:
            raise ValueError("corpus must contain at least one token")

    def generate(self, seed_prompt: str, max_new_tokens: int = 20) -> str:
        seed = seed_prompt.strip()
        if not seed:
            raise ValueError("seed_prompt must contain at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")

        tokens = seed.split()
        generated: list[str] = []
        for _ in range(max_new_tokens):
            if tokens[-1].endswith((".", "!", "?")):
                break

            next_token = self._next_token(tokens)
            if next_token is None:
                break

            tokens.append(next_token)
            generated.append(next_token)

        return seed if not generated else f"{seed} {' '.join(generated)}"

    def _next_token(self, tokens: list[str]) -> str | None:
        for context_size in range(min(self.n - 1, len(tokens)), -1, -1):
            context = tuple(tokens[-context_size:]) if context_size else ()
            continuations = self._continuations.get(context)
            if continuations:
                return continuations.most_common(1)[0][0]
        return None


# --- Milestone 2: Implement the n-gram model and generate text from it.
# Hint: Apply the N-gram model technique from the 'Build a Next-Word Predictor from Scratch' drill.
def milestone_2() -> NGramModel:
    return NGramModel(milestone_1())


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Module, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be at least 1")
        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")

        weight = getattr(base_layer, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError("LoRA targets must have a two-dimensional weight")
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "Conv1D":
            in_features, out_features = weight.shape
        else:
            raise TypeError("LoRA targets must be nn.Linear or Transformers Conv1D layers")

        self.base_layer = base_layer
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_A.to(device=weight.device, dtype=weight.dtype)
        self.lora_B.to(device=weight.device, dtype=weight.dtype)
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / rank

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_layer(inputs) + self.lora_B(self.lora_A(inputs)) * self.scaling


def inject_lora_adapters(
    model: nn.Module,
    target_modules: tuple[str, ...] = ("c_attn", "c_proj"),
    rank: int = 4,
    alpha: float = 8.0,
) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    injected = 0
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if child_name in target_modules:
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                injected += 1

    if injected == 0:
        raise ValueError(f"No LoRA target layers found for {target_modules}")
    return model


# --- Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
# Hint: Use the 'Build LoRA from Scratch' drill's technique to prepare the LLM.
def milestone_3() -> tuple[nn.Module, PreTrainedTokenizerBase]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return inject_lora_adapters(model), tokenizer


# --- Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
# Hint: Train the LoRA adapter on your collected sports commentary corpus.
def milestone_4() -> dict[str, torch.Tensor]:
    model, tokenizer = milestone_3()
    corpus = milestone_1()
    if not corpus:
        raise ValueError("training corpus must contain at least one line")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer needs an EOS token to pad training batches")
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        corpus,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    if input_ids.ndim != 2 or input_ids.shape[0] == 0:
        raise ValueError("tokenizer must produce a non-empty batch of token sequences")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    trainable_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("model has no trainable LoRA adapter weights")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3)
    batch_size = 4
    for start in range(0, input_ids.shape[0], batch_size):
        batch_ids = input_ids[start : start + batch_size]
        batch_mask = attention_mask[start : start + batch_size]
        labels = batch_ids.clone()
        labels[batch_mask == 0] = -100

        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=batch_ids, attention_mask=batch_mask, labels=labels)
        loss = getattr(output, "loss", None)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("causal language model must return a tensor loss")
        if not torch.isfinite(loss).all():
            raise ValueError("training produced a non-finite loss")
        loss.backward()
        optimizer.step()

    model.eval()
    adapter_weights = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if "lora_" in name
    }
    if not adapter_weights:
        raise RuntimeError("model has no LoRA adapter weights to save")

    ADAPTER_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_weights, ADAPTER_WEIGHTS_PATH)
    return adapter_weights


# --- Milestone 5: Generate text from the LoRA-adapted LLM.
# Hint: Generate text using your fine-tuned LoRA-adapted LLM.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
# Hint: Design a simple interface to display and compare the generated texts.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
