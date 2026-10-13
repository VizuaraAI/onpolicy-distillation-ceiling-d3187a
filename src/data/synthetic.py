"""Generators for the ABSENT capabilities.

These tasks are generated rather than scraped. That is deliberate: the
whole experiment rests on being able to assert that the base student
never saw the symbol system, and no natural dataset lets us assert that.

Every generator returns ``Example`` records with:

- ``prompt``: what the model sees
- ``answer``: the exact-match target
- ``variables``: task-critical intermediate variables, used by the
  representability audit as probe targets (for example the carry bit)

Every task also exposes a verifier so RL with verifiable reward and
pass@k scoring use exactly the same grading function.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Callable

# Digit glyphs for base-7 arithmetic. Deliberately not 0-9, so the
# student cannot fall back on memorized base-10 digit semantics.
BASE7_GLYPHS = ["@", "#", "$", "%", "&", "~", "^"]

# A script the tokenizer is unlikely to represent densely. Replace this
# with any target script you prefer; keep it fixed across runs.
TRANSLIT_TABLE = {
    "a": "ɑ", "b": "β", "c": "ç", "d": "ð", "e": "ɛ",
    "f": "ɸ", "g": "ɣ", "h": "ħ", "i": "ɩ", "j": "ʝ",
    "k": "ĸ", "l": "ɭ", "m": "ɱ", "n": "ŋ", "o": "ɔ",
    "p": "ρ", "q": "ʠ", "r": "ɾ", "s": "ʃ", "t": "ʈ",
    "u": "ʊ", "v": "ʋ", "w": "ʍ", "x": "χ", "y": "ʎ",
    "z": "ʐ",
}


@dataclass
class Example:
    """One graded item from the battery."""

    prompt: str
    answer: str
    variables: dict[str, int | str] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)


def exact_match(prediction: str, answer: str) -> float:
    """Strict verifier, after whitespace normalization."""
    return float(prediction.strip() == answer.strip())


def char_overlap(prediction: str, answer: str) -> float:
    """Partial credit: fraction of answer positions matched in order."""
    pred, gold = prediction.strip(), answer.strip()
    if not gold:
        return 0.0
    matched = sum(1 for p, g in zip(pred, gold) if p == g)
    return matched / len(gold)


def _to_base(value: int, base: int, glyphs: list[str]) -> str:
    if value == 0:
        return glyphs[0]
    digits: list[str] = []
    while value:
        digits.append(glyphs[value % base])
        value //= base
    return "".join(reversed(digits))


def base7_arith_glyphs(n: int, rng: random.Random) -> list[Example]:
    """Addition in base 7 written with non-standard digit glyphs.

    Probe targets: whether a carry occurs, and the number of carries.
    These are the intermediate variables the student would need to
    represent in order to do the task at all.
    """
    legend = ", ".join(f"{g}={i}" for i, g in enumerate(BASE7_GLYPHS))
    examples: list[Example] = []
    for _ in range(n):
        a = rng.randint(1, 7 ** 3 - 1)
        b = rng.randint(1, 7 ** 3 - 1)
        total = a + b
        n_carries = 0
        x, y = a, b
        carry = 0
        while x or y or carry:
            s = (x % 7) + (y % 7) + carry
            carry = 1 if s >= 7 else 0
            n_carries += carry
            x //= 7
            y //= 7
        prompt = (
            "You are working in base 7 with these digit symbols: "
            f"{legend}.\n"
            f"Compute {_to_base(a, 7, BASE7_GLYPHS)} + "
            f"{_to_base(b, 7, BASE7_GLYPHS)} and write the result "
            "using the same symbols.\nAnswer: "
        )
        examples.append(
            Example(
                prompt=prompt,
                answer=_to_base(total, 7, BASE7_GLYPHS),
                variables={"any_carry": int(n_carries > 0), "n_carries": n_carries},
                meta={"a": a, "b": b},
            )
        )
    return examples


def script_transliteration(n: int, rng: random.Random) -> list[Example]:
    """Transliterate an English word into a held-out glyph inventory.

    Probe target: the identity of the first target grapheme, which the
    student must represent before it can emit it.
    """
    vocab = [
        "research", "distill", "feature", "student", "teacher", "policy",
        "ceiling", "latent", "sparse", "reward", "probe", "signal",
        "model", "update", "rollout", "budget", "gradient", "encoder",
    ]
    examples: list[Example] = []
    key_lines = ", ".join(f"{k}->{v}" for k, v in list(TRANSLIT_TABLE.items())[:8])
    for _ in range(n):
        word = rng.choice(vocab)
        target = "".join(TRANSLIT_TABLE[ch] for ch in word)
        prompt = (
            "Transliterate the word into the target script. "
            f"Partial key: {key_lines}, and so on for the rest of the "
            "alphabet.\n"
            f"Word: {word}\nTransliteration: "
        )
        examples.append(
            Example(
                prompt=prompt,
                answer=target,
                variables={"first_grapheme": target[0], "length": len(word)},
                meta={"word": word},
            )
        )
    return examples


def substitution_cipher(n: int, rng: random.Random, key_seed: int = 1234) -> list[Example]:
    """Decode a substitution cipher under one fixed, unseen key.

    The key is fixed by ``key_seed`` so it is identical across every
    arm and every seed. Probe target: the plaintext first letter.
    """
    key_rng = random.Random(key_seed)
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    key_rng.shuffle(shuffled)
    encode = dict(zip(letters, shuffled))
    words = [
        "ceiling", "feature", "reweight", "capacity", "transfer",
        "evidence", "baseline", "ablation", "control", "measure",
    ]
    examples: list[Example] = []
    hint = ", ".join(f"{k}->{encode[k]}" for k in letters[:6])
    for _ in range(n):
        plain = rng.choice(words)
        cipher = "".join(encode[ch] for ch in plain)
        prompt = (
            "A fixed substitution cipher was used. Partial key: "
            f"{hint}, and so on.\n"
            f"Ciphertext: {cipher}\nPlaintext: "
        )
        examples.append(
            Example(
                prompt=prompt,
                answer=plain,
                variables={"first_letter": plain[0], "length": len(plain)},
                meta={"cipher": cipher},
            )
        )
    return examples


def synthetic_grammar_validity(n: int, rng: random.Random) -> list[Example]:
    """Decide whether a string is in a small generated formal language.

    The language is balanced bracket sequences over three bracket types
    with a depth cap, which the student is unlikely to have a dedicated
    representation for. Probe target: the maximum nesting depth and
    validity.
    """
    pairs = [("<", ">"), ("[", "]"), ("{", "}")]

    def gen_valid(depth_cap: int) -> str:
        if depth_cap <= 0 or rng.random() < 0.3:
            return ""
        left, right = rng.choice(pairs)
        return left + gen_valid(depth_cap - 1) + right + gen_valid(depth_cap - 1)

    def max_depth(s: str) -> int:
        depth = best = 0
        openers = {p[0] for p in pairs}
        for ch in s:
            if ch in openers:
                depth += 1
                best = max(best, depth)
            else:
                depth -= 1
        return best

    examples: list[Example] = []
    for _ in range(n):
        s = gen_valid(4) or "<>"
        valid = True
        if rng.random() < 0.5 and len(s) >= 2:
            # Corrupt by swapping two characters, which usually breaks it.
            chars = list(s)
            i, j = rng.sample(range(len(chars)), 2)
            chars[i], chars[j] = chars[j], chars[i]
            candidate = "".join(chars)
            if candidate != s:
                s = candidate
                valid = _is_balanced(s, pairs)
        prompt = (
            "A string is VALID if every bracket is correctly nested and "
            "closed, using only <>, [] and {}.\n"
            f"String: {s}\nAnswer (VALID or INVALID): "
        )
        examples.append(
            Example(
                prompt=prompt,
                answer="VALID" if valid else "INVALID",
                variables={"valid": int(valid), "max_depth": max_depth(s)},
                meta={"string": s},
            )
        )
    return examples


def _is_balanced(s: str, pairs: list[tuple[str, str]]) -> bool:
    closer_of = dict(pairs)
    openers = set(closer_of)
    stack: list[str] = []
    for ch in s:
        if ch in openers:
            stack.append(ch)
        else:
            if not stack or closer_of[stack.pop()] != ch:
                return False
    return not stack


# name -> (generator, verifier, partial-credit scorer)
ABSENT_GENERATORS: dict[str, tuple[Callable[..., list[Example]], Callable[[str, str], float], Callable[[str, str], float]]] = {
    "base7_arith_glyphs": (base7_arith_glyphs, exact_match, char_overlap),
    "script_transliteration": (script_transliteration, exact_match, char_overlap),
    "substitution_cipher": (substitution_cipher, exact_match, char_overlap),
    "synthetic_grammar_validity": (synthetic_grammar_validity, exact_match, exact_match),
}
