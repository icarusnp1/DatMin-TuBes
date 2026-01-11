"""Porter Stemmer for Bahasa Indonesia (Snowball-style rules)

This module implements (in Python) the Snowball script you provided.
It is intentionally self-contained so you can swap this single file to
change stemming behavior without touching other modules.

Public API:
    stem_word(word: str) -> str
    stem_tokens(tokens: list[str]) -> list[str]
"""

from __future__ import annotations

from dataclasses import dataclass

VOWELS = set("aeiou")

def _count_vowels(s: str) -> int:
    return sum(1 for ch in s if ch in VOWELS)

@dataclass
class StemState:
    measure: int
    prefix_code: int  # 0 none, 1 di/meng/ter, 2 per, 3 ke/peng, 4 ber

def _remove_particle(word: str, st: StemState) -> str:
    # backward among: kah lah pun
    for suf in ("kah", "lah", "pun"):
        if word.endswith(suf) and st.measure > 2:
            word = word[:-len(suf)]
            st.measure -= 1
            break
    return word

def _remove_possessive_pronoun(word: str, st: StemState) -> str:
    # backward among: ku mu nya
    for suf in ("ku", "mu", "nya"):
        if word.endswith(suf) and st.measure > 2:
            word = word[:-len(suf)]
            st.measure -= 1
            break
    return word

def _remove_first_order_prefix(word: str, st: StemState) -> tuple[str, bool]:
    """Remove first-order derivational prefix.

    Returns (new_word, removed?).
    Sets st.prefix_code and decrements st.measure by 1 when removal happens.
    """
    w = word

    # di, meng, me, ter -> code 1
    for p in ("di", "meng", "me", "ter"):
        if w.startswith(p) and len(w) > len(p):
            st.prefix_code = 1
            st.measure -= 1
            return w[len(p):], True

    # men- special: menyV -> sV..., else delete men-
    if w.startswith("meny") and len(w) >= 5 and w[4] in VOWELS:
        st.prefix_code = 1
        st.measure -= 1
        return "s" + w[4:], True
    if w.startswith("men") and len(w) > 3:
        st.prefix_code = 1
        st.measure -= 1
        return w[3:], True

    # ke, peng -> code 3
    for p in ("ke", "peng"):
        if w.startswith(p) and len(w) > len(p):
            st.prefix_code = 3
            st.measure -= 1
            return w[len(p):], True

    # pen- special: penyV -> sV..., else delete pen-
    if w.startswith("peny") and len(w) >= 5 and w[4] in VOWELS:
        st.prefix_code = 3
        st.measure -= 1
        return "s" + w[4:], True
    if w.startswith("pen") and len(w) > 3:
        st.prefix_code = 3
        st.measure -= 1
        return w[3:], True

    # mem-: if memV -> pV..., else delete mem-
    if w.startswith("mem") and len(w) > 3:
        st.prefix_code = 1
        st.measure -= 1
        rest = w[3:]
        if rest and rest[0] in VOWELS:
            return "p" + rest, True
        return rest, True

    # pem-: if pemV -> pV..., else delete pem-
    if w.startswith("pem") and len(w) > 3:
        st.prefix_code = 3
        st.measure -= 1
        rest = w[3:]
        if rest and rest[0] in VOWELS:
            return "p" + rest, True
        return rest, True

    return word, False

def _remove_second_order_prefix(word: str, st: StemState) -> tuple[str, bool]:
    """Remove second-order derivational prefix (pe-/be- families).

    Returns (new_word, removed?).
    Per script note: all prefixes removed here contain exactly one vowel, so st.measure -= 1 on removal.
    """
    w = word

    # pe-
    if w.startswith("pelajar"):
        # special case: pelajar..., pelajaran... -> ajar...
        st.prefix_code = 2
        st.measure -= 1
        return w[3:], True  # remove 'pel'
    if w.startswith("per") and len(w) > 3:
        st.prefix_code = 2
        st.measure -= 1
        return w[3:], True
    if w.startswith("pe") and len(w) > 2:
        st.prefix_code = 2
        st.measure -= 1
        return w[2:], True

    # be-
    if w.startswith("belajar"):
        st.prefix_code = 4
        st.measure -= 1
        return w[3:], True  # remove 'bel'
    if w.startswith("ber") and len(w) > 3:
        st.prefix_code = 4
        st.measure -= 1
        return w[3:], True
    # be + C + er...
    if w.startswith("be") and len(w) >= 5 and (w[2] not in VOWELS) and w[3:5] == "er":
        st.prefix_code = 4
        st.measure -= 1
        return w[2:], True  # remove 'be'
    # fallback: be-
    if w.startswith("be") and len(w) > 2:
        st.prefix_code = 4
        st.measure -= 1
        return w[2:], True

    return word, False

def _remove_suffix(word: str, st: StemState) -> tuple[str, bool]:
    """Remove derivational suffixes (-i, -an, -kan) with Snowball-script conditions."""
    w = word
    if st.measure <= 2:
        return w, False

    # Handle ...kan vs ...an logic using the script's conditions (with SUFFIX_KAN_NOTE amendment)
    if w.endswith("kan") and len(w) > 3:
        # remove 'kan' only if prefix not in {ke/peng (3), per (2)}
        if st.prefix_code not in (2, 3):
            st.measure -= 1
            return w[:-3], True
        # otherwise, the script can fall back to removing 'an' if prefix != 1
        if st.prefix_code != 1:
            st.measure -= 1
            return w[:-2], True
        return w, False

    if w.endswith("an") and len(w) > 2:
        # remove 'an' if prefix != 1
        if st.prefix_code != 1:
            st.measure -= 1
            return w[:-2], True
        return w, False

    # Suffix -i: allowed if prefix <= 2 and word does not end -si
    if w.endswith("i") and len(w) > 1:
        if st.prefix_code <= 2 and (not w.endswith("si")):
            st.measure -= 1
            return w[:-1], True
        return w, False

    return w, False

def stem_word(word: str) -> str:
    """Stem a single token (expects lowercase alphabetic Indonesian token)."""
    if not word:
        return word

    st = StemState(measure=_count_vowels(word), prefix_code=0)

    # Gate: only stem words with measure > 2
    if st.measure <= 2:
        return word

    # Backward removals: particle then possessive pronoun
    word2 = _remove_particle(word, st)
    if st.measure <= 2:
        return word2
    word2 = _remove_possessive_pronoun(word2, st)
    if st.measure <= 2:
        return word2

    # Prefix and suffix workflow (mirrors the Snowball stem routine)
    st.prefix_code = 0
    w3, removed_first = _remove_first_order_prefix(word2, st)
    if removed_first:
        if st.measure > 2:
            w3, _ = _remove_suffix(w3, st)
        if st.measure > 2:
            w3, _ = _remove_second_order_prefix(w3, st)
        return w3

    # If first-order didn't apply:
    w4, removed_second = _remove_second_order_prefix(word2, st)
    if removed_second and st.measure > 2:
        w4, _ = _remove_suffix(w4, st)
    return w4

def stem_tokens(tokens: list[str]) -> list[str]:
    return [stem_word(t) for t in tokens]
