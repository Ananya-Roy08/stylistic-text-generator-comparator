"""Your code. One stub per milestone; build them in order and keep each one small."""

from collections import Counter, defaultdict
from pathlib import Path


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


# --- Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
# Hint: Use the 'Build LoRA from Scratch' drill's technique to prepare the LLM.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
# Hint: Train the LoRA adapter on your collected sports commentary corpus.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Generate text from the LoRA-adapted LLM.
# Hint: Generate text using your fine-tuned LoRA-adapted LLM.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Provide a qualitative comparison interface for generated texts from both models.
# Hint: Design a simple interface to display and compare the generated texts.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
