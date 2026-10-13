"""Public dataset loading for the REPRESENTED capabilities.

Nothing large is committed to this repository. Everything here
downloads from the Hugging Face hub on first use and is cached under
``HF_HOME``.
"""

from __future__ import annotations

import logging
import random
import re

from .synthetic import Example, char_overlap, exact_match

logger = logging.getLogger(__name__)

_FORMAT_INSTRUCTIONS = [
    ("Answer with exactly three comma separated words.", "three_words"),
    ("Answer in uppercase only.", "uppercase"),
    ("Answer as a single integer with no other text.", "bare_integer"),
    ("Answer as a JSON object with the single key \"answer\".", "json_key"),
]


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "pip install 'datasets>=3.0.0' to load the REPRESENTED tasks"
        ) from exc
    return load_dataset


def _final_number(text: str) -> str:
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return matches[-1].replace(",", "") if matches else ""


def number_match(prediction: str, answer: str) -> float:
    """Verifier for numeric answers: compare the last number emitted."""
    return float(_final_number(prediction) == _final_number(answer))


def gsm8k_math(n: int, rng: random.Random, split: str = "train") -> list[Example]:
    """Grade-school math word problems.

    Probe targets are coarse here on purpose: the audit only needs a
    task-critical variable that the student plausibly represents, and
    the number of reasoning steps plus answer magnitude serve that role.
    """
    load_dataset = _require_datasets()
    ds = load_dataset("openai/gsm8k", "main", split=split)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    examples: list[Example] = []
    for idx in indices[:n]:
        row = ds[idx]
        solution = row["answer"]
        gold = solution.split("####")[-1].strip()
        n_steps = solution.count("\n") + 1
        examples.append(
            Example(
                prompt=f"{row['question']}\nAnswer: ",
                answer=gold,
                variables={
                    "n_steps": n_steps,
                    "answer_magnitude": len(_final_number(gold)),
                },
                meta={"solution": solution},
            )
        )
    return examples


def base10_multistep_arith(n: int, rng: random.Random) -> list[Example]:
    """Multi-step base-10 arithmetic, the in-distribution counterpart
    to the base-7 ABSENT task. Same structure, familiar symbols."""
    examples: list[Example] = []
    for _ in range(n):
        a, b, c = (rng.randint(11, 99) for _ in range(3))
        total = a + b * c
        n_carries = sum(
            1
            for x, y in [(a % 10, (b * c) % 10)]
            if x + y >= 10
        )
        examples.append(
            Example(
                prompt=f"Compute {a} + {b} * {c}.\nAnswer: ",
                answer=str(total),
                variables={"any_carry": n_carries, "n_digits": len(str(total))},
                meta={"a": a, "b": b, "c": c},
            )
        )
    return examples


def format_compliance(n: int, rng: random.Random) -> list[Example]:
    """Instruction and output-format compliance.

    The clearest REPRESENTED capability: small instruct models already
    carry format features, so this is where reweighting should shine.
    """
    facts = [
        ("What is the capital of France?", "Paris"),
        ("How many sides does a hexagon have?", "6"),
        ("What colour is the sky on a clear day?", "blue"),
        ("What is two plus three?", "5"),
        ("Name the largest ocean.", "Pacific"),
    ]
    examples: list[Example] = []
    for _ in range(n):
        question, answer = rng.choice(facts)
        instruction, kind = rng.choice(_FORMAT_INSTRUCTIONS)
        if kind == "uppercase":
            target = answer.upper()
        elif kind == "json_key":
            target = '{"answer": "%s"}' % answer
        elif kind == "bare_integer" and answer.isdigit():
            target = answer
        else:
            target = answer
        examples.append(
            Example(
                prompt=f"{instruction}\n{question}\n",
                answer=target,
                variables={"format_kind": kind},
                meta={"instruction": instruction},
            )
        )
    return examples


def extractive_qa(n: int, rng: random.Random, split: str = "train") -> list[Example]:
    """Short-form extractive QA over SQuAD contexts."""
    load_dataset = _require_datasets()
    ds = load_dataset("rajpurkar/squad", split=split)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    examples: list[Example] = []
    for idx in indices[:n]:
        row = ds[idx]
        answers = row["answers"]["text"]
        if not answers:
            continue
        gold = answers[0]
        examples.append(
            Example(
                prompt=(
                    f"Context: {row['context']}\n"
                    f"Question: {row['question']}\n"
                    "Answer with a span copied from the context.\nAnswer: "
                ),
                answer=gold,
                variables={"answer_tokens": len(gold.split())},
                meta={"title": row.get("title", "")},
            )
        )
    return examples


REPRESENTED_GENERATORS = {
    "gsm8k_math": (gsm8k_math, number_match, number_match),
    "base10_multistep_arith": (base10_multistep_arith, number_match, number_match),
    "format_compliance": (format_compliance, exact_match, char_overlap),
    "extractive_qa": (extractive_qa, exact_match, char_overlap),
}
