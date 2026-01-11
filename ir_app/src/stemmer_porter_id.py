"""Porter-style Indonesian stemmer (simplified, swappable module).

The user can replace this file only, as long as these functions exist:
  - stem_word(word: str) -> str
  - stem_tokens(tokens: list[str]) -> list[str]

NOTE: This is a pragmatic implementation for IR coursework; it covers common
Indonesian affixes and particles.
"""

from __future__ import annotations
import re

_PARTICLES = ("kah", "lah", "pun")
_PRONOUNS = ("ku", "mu", "nya")

_PREFIXES_1 = (
    "meng", "meny", "men", "mem", "me",
    "peng", "peny", "pen", "pem", "di", "ter", "ke"
)
_PREFIXES_2 = ("ber", "bel", "be", "per", "pel", "pe")

_SUFFIXES = ("kan", "an", "i")

_VOWELS = set("aeiou")

def _strip_particle(w: str) -> str:
    for p in _PARTICLES:
        if w.endswith(p) and len(w) > len(p) + 2:
            return w[:-len(p)]
    return w

def _strip_pronoun(w: str) -> str:
    for p in _PRONOUNS:
        if w.endswith(p) and len(w) > len(p) + 2:
            return w[:-len(p)]
    return w

def _strip_suffix(w: str) -> str:
    # Protect "-si" words (televisi, organisasi, komunikasi)
    if w.endswith("i") and w.endswith("si"):
        return w
    for s in _SUFFIXES:
        if w.endswith(s) and len(w) > len(s) + 2:
            return w[:-len(s)]
    return w

def _strip_prefix(w: str) -> str:
    # First-order
    for p in sorted(_PREFIXES_1, key=len, reverse=True):
        if w.startswith(p) and len(w) > len(p) + 2:
            stem = w[len(p):]
            # handle meny- -> s-
            if p == "meny" and stem and stem[0] in _VOWELS:
                return "s" + stem
            # mem-/pem- -> p- if followed by vowel
            if p in ("mem", "pem") and stem and stem[0] in _VOWELS:
                return "p" + stem
            # men-/pen- -> t- if followed by vowel (rough heuristic)
            if p in ("men", "pen") and stem and stem[0] in _VOWELS:
                return "t" + stem
            return stem
    # Second-order
    for p in sorted(_PREFIXES_2, key=len, reverse=True):
        if w.startswith(p) and len(w) > len(p) + 2:
            stem = w[len(p):]
            # bel-/pel- + ajar...
            if p in ("bel", "pel") and stem.startswith("ajar"):
                return "ajar"
            return stem
    return w

def stem_word(word: str) -> str:
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) <= 3:
        return w
    w = _strip_particle(w)
    w = _strip_pronoun(w)
    w2 = _strip_prefix(w)
    # suffix after prefix stripping often helps
    w2 = _strip_suffix(w2)
    # second pass suffix stripping for robustness
    w2 = _strip_suffix(w2)
    return w2

def stem_tokens(tokens: list[str]) -> list[str]:
    return [stem_word(t) for t in tokens]
