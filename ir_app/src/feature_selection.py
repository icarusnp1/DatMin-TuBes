from __future__ import annotations
from typing import Dict, List, Set, Tuple
import math

def compute_df(doc_tokens: Dict[str, List[str]]) -> Dict[str, int]:
    df: Dict[str, int] = {}
    for _, toks in doc_tokens.items():
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return df

def select_features_df(
    doc_tokens: Dict[str, List[str]],
    *,
    min_df: int = 2,
    max_df_ratio: float = 0.85,
    top_n: int | None = 8000,
) -> Tuple[Set[str], dict]:
    N = len(doc_tokens)
    df = compute_df(doc_tokens)
    if N == 0:
        return set(), {"N": 0, "vocab": 0, "selected": 0}

    max_df = max(1, int(math.floor(max_df_ratio * N)))
    kept = [(t, c) for t, c in df.items() if c >= min_df and c <= max_df]
    kept.sort(key=lambda x: (-x[1], x[0]))

    if top_n is not None and len(kept) > top_n:
        kept = kept[:top_n]

    selected = {t for t, _ in kept}
    report = {
        "method": "DF threshold + Top-N",
        "N": N,
        "vocab": len(df),
        "selected": len(selected),
        "min_df": min_df,
        "max_df_ratio": max_df_ratio,
        "max_df": max_df,
        "top_n": top_n,
    }
    return selected, report
