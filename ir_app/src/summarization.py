from __future__ import annotations
from typing import Dict, List, Tuple
import re
from .preprocessing import preprocess

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}|\r\n{2,}|\n|\r")

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    parts = [p for p in parts if len(p) >= 20]
    return parts

def summarize_extractive(text: str, *, idf: Dict[str, float], num_sentences: int = 2, max_chars: int = 350) -> str:
    sents = split_sentences(text)
    if not sents:
        return ""
    scored: List[Tuple[int, float]] = []
    for i, sent in enumerate(sents):
        terms = preprocess(sent)
        score = sum(idf.get(t, 0.0) for t in terms) if terms else 0.0
        scored.append((i, score))
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:max(1, num_sentences)]
    top_idx = sorted(i for i, _ in top)
    summary = " ".join(sents[i] for i in top_idx)
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
    return summary
