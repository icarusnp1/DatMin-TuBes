"""CLI app: build index, run query, show ranking + summaries."""

from __future__ import annotations

import argparse
import os
import json

from .preprocessing import preprocess
from .indexing import build_inverted_index, save_index, load_index, compute_idf
from .retrieval import TfidfModel, build_doc_vectors, build_query_vector, retrieve
from .summarization import summarize_extractive

def load_documents(data_dir: str) -> dict:
    docs = {}
    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith(".txt"):
            continue
        doc_id = fn
        with open(os.path.join(data_dir, fn), "r", encoding="utf-8") as f:
            docs[doc_id] = f.read()
    return docs

def cmd_build(args: argparse.Namespace) -> None:
    docs = load_documents(args.data_dir)
    if not docs:
        raise SystemExit(f"No .txt files found in {args.data_dir}")

    doc_tokens = {doc_id: preprocess(text) for doc_id, text in docs.items()}
    index = build_inverted_index(doc_tokens)
    save_index(index, args.index_path)

    idf = compute_idf(index, smooth=True)
    with open(args.idf_path, "w", encoding="utf-8") as f:
        json.dump(idf, f, ensure_ascii=False)

    print(f"Built index for N={index.N} documents") 
    print(f"Saved index: {args.index_path}")
    print(f"Saved idf:   {args.idf_path}")

def cmd_query(args: argparse.Namespace) -> None:
    docs = load_documents(args.data_dir)
    index = load_index(args.index_path)
    with open(args.idf_path, "r", encoding="utf-8") as f:
        idf = {k: float(v) for k, v in json.load(f).items()}

    model = TfidfModel(idf=idf)
    # Build doc vectors once for querying session
    doc_tokens = {doc_id: preprocess(text) for doc_id, text in docs.items()}
    doc_vecs = build_doc_vectors(index, model)

    q_terms = preprocess(args.query)
    q_vec = build_query_vector(q_terms, model)
    ranked = retrieve(q_vec, index, doc_vecs, top_k=args.top_k)

    if not ranked:
        print("No results.")
        return

    print(f"Query: {args.query}")
    print("\nResults:")
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        summary = summarize_extractive(docs[doc_id], idf=idf, num_sentences=args.summary_sentences)
        print(f"{rank:>2}) {doc_id}   score={score:.4f}")
        if summary:
            # print(f"    summary: {summary}")
            print("")

def main():
    p = argparse.ArgumentParser(description="Simple Indonesian IR System (TF-IDF + cosine) with Porter(ID) stemming.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build inverted index + idf")
    p_build.add_argument("--data-dir", default="data/raw_txt", help="Directory with .txt documents")
    p_build.add_argument("--index-path", default="artifacts/index.json", help="Path to save index JSON")
    p_build.add_argument("--idf-path", default="artifacts/idf.json", help="Path to save idf JSON")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Run a query and print ranking + summaries")
    p_query.add_argument("--data-dir", default="data/raw_txt", help="Directory with .txt documents")
    p_query.add_argument("--index-path", default="artifacts/index.json", help="Path to index JSON")
    p_query.add_argument("--idf-path", default="artifacts/idf.json", help="Path to idf JSON")
    p_query.add_argument("--query", required=True, help="Query string")
    p_query.add_argument("--top-k", type=int, default=10)
    p_query.add_argument("--summary-sentences", type=int, default=2)
    p_query.set_defaults(func=cmd_query)

    args = p.parse_args()
    os.makedirs("artifacts", exist_ok=True)
    args.func(args)

if __name__ == "__main__":
    main()
