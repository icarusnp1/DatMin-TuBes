from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import json
import math

# Inverted index: term -> {doc_id: tf}
InvertedIndex = Dict[str, Dict[str, int]]

def build_inverted_index(doc_tokens: Dict[str, List[str]], allowed_terms: set[str] | None = None) -> InvertedIndex:
    index: InvertedIndex = {}
    for doc_id, toks in doc_tokens.items():
        tf_map: Dict[str, int] = {}
        for t in toks:
            if allowed_terms is not None and t not in allowed_terms:
                continue
            tf_map[t] = tf_map.get(t, 0) + 1
        for t, tf in tf_map.items():
            postings = index.setdefault(t, {})
            postings[doc_id] = tf
    return index

def compute_idf(index: InvertedIndex, *, smooth: bool = True) -> Dict[str, float]:
    # N is number of docs; infer from postings union
    docs = set()
    for postings in index.values():
        docs.update(postings.keys())
    N = max(1, len(docs))
    idf: Dict[str, float] = {}
    for term, postings in index.items():
        df = len(postings)
        if smooth:
            idf[term] = math.log((1 + N) / (1 + df)) + 1.0
        else:
            idf[term] = math.log(N / max(1, df))
    return idf

def save_index(index: InvertedIndex, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

def load_index(path: str) -> InvertedIndex:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_tf_df_idf_matrix(
    *,
    index: InvertedIndex,
    idf: Dict[str, float],
    doc_ids: List[str],
    terms: Optional[List[str]] = None,
    q_tf: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Matriks:
      Term | D1 | D2 | ... | Dk | (Q) | DF | IDF

    - terms=None => semua term (sorted)
    - q_tf diberikan => tambah kolom Q (TF query)
    """
    if terms is None:
        terms = sorted(index.keys())

    cols = list(doc_ids)
    has_q = q_tf is not None
    if has_q:
        cols = cols + ["Q"]

    rows = []
    for term in terms:
        postings = index.get(term, {}) or {}
        tf_docs = [int(postings.get(d, 0)) for d in doc_ids]
        if has_q:
            tf_docs.append(int((q_tf or {}).get(term, 0)))
        df = int(len(postings))
        rows.append(
            {
                "term": term,
                "tfs": tf_docs,
                "df": df,
                "idf": float(idf.get(term, 0.0)),
            }
        )

    return {"cols": cols, "rows": rows}