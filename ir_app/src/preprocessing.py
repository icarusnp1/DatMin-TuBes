"""Text preprocessing: normalize -> tokenize -> stopword removal -> stemming."""

from __future__ import annotations

import re
from typing import Iterable, List, Set
from .stemmer_porter_id import stem_tokens

_TOKEN_RE = re.compile(r"[a-zA-Z]+", re.UNICODE)

DEFAULT_STOPWORDS: Set[str] = {
    # Minimal Indonesian stopword set (extend as needed)
    "yang","dan","di","ke","dari","pada","untuk","dengan","atau","itu","ini","sebagai",
    "oleh","dalam","adalah","akan","tidak","bukan","sudah","belum","lebih","juga","karena",
    "ada","saat","sehingga","agar","maka","jadi","pula","hanya","para","dapat","bisa","harus"
}

def normalize_text(text: str) -> str:
    text = text.lower()
    # Keep letters; replace other chars with space to avoid token concatenation
    text = re.sub(r"[^a-zA-Z\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

def remove_stopwords(tokens: Iterable[str], stopwords: Set[str] | None = None) -> List[str]:
    sw = DEFAULT_STOPWORDS if stopwords is None else stopwords
    return [t for t in tokens if t and t not in sw]

def preprocess(text: str, *, stopwords: Set[str] | None = None, do_stem: bool = True) -> List[str]:
    norm = normalize_text(text)
    toks = tokenize(norm)
    toks = remove_stopwords(toks, stopwords=stopwords)
    if do_stem:
        toks = stem_tokens(toks)
    return toks
