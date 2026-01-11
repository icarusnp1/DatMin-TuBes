from __future__ import annotations
import os
from typing import Dict

def load_txt_dir(txt_dir: str) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    if not os.path.isdir(txt_dir):
        return docs
    for fn in sorted(os.listdir(txt_dir)):
        if fn.lower().endswith(".txt"):
            path = os.path.join(txt_dir, fn)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs[fn] = f.read()
    return docs

def extract_text_from_pdf(pdf_path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    parts = [page.get_text() for page in doc]
    return "\n".join(parts)

def ingest_pdf_dir(pdf_dir: str, out_txt_dir: str) -> Dict[str, str]:
    os.makedirs(out_txt_dir, exist_ok=True)
    docs: Dict[str, str] = {}
    if not os.path.isdir(pdf_dir):
        return docs
    for fn in sorted(os.listdir(pdf_dir)):
        if not fn.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_dir, fn)
        txt_fn = os.path.splitext(fn)[0] + ".txt"
        txt_path = os.path.join(out_txt_dir, txt_fn)
        text = extract_text_from_pdf(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        docs[txt_fn] = text
    return docs

def load_corpus(*, txt_dir: str, pdf_dir: str, ingested_txt_dir: str) -> Dict[str, str]:
    docs = {}
    docs.update(load_txt_dir(txt_dir))
    pdf_docs = ingest_pdf_dir(pdf_dir, ingested_txt_dir)
    for k, v in pdf_docs.items():
        docs.setdefault(k, v)
    return docs
