"""Your code. One stub per milestone; build them in order and keep each one small."""

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


# --- Milestone 2: Implement the n-gram model and generate text from it.
# Hint: Apply the N-gram model technique from the 'Build a Next-Word Predictor from Scratch' drill.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


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
