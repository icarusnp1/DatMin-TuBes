"""Extractive summarization for Indonesian text (simple TF-IDF sentence scoring)."""

from __future__ import annotations

from typing import Dict, List, Tuple
import re

from .preprocessing import preprocess

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sents = _SENT_SPLIT.split(text)
    # fallback if no punctuation
    return [s.strip() for s in sents if s.strip()]

def summarize_extractive(
    text: str,
    *,
    idf: Dict[str, float],
    num_sentences: int = 2,
) -> str:
    sents = split_sentences(text)
    if not sents:
        return ""

    scored: List[Tuple[int, float]] = []
    for i, sent in enumerate(sents):
        terms = preprocess(sent)  # same pipeline (incl stemming)
        if not terms:
            scored.append((i, 0.0))
            continue
        score = 0.0
        for t in terms:
            score += idf.get(t, 0.0)
        scored.append((i, score))

    # pick top sentences, then restore original order
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:max(1, num_sentences)]
    top_idx = sorted(i for i, _ in top)
    return " ".join(sents[i] for i in top_idx)
