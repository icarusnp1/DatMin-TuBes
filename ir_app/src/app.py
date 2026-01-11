from __future__ import annotations
import argparse, os, json

from .loaders import load_txt_dir
from .preprocessing import preprocess
from .feature_selection import select_features_df
from .indexing import build_inverted_index, compute_idf, save_index, load_index
from .retrieval import TfidfModel, build_doc_vectors, build_query_vector, retrieve
from .summarization import summarize_extractive

ART_DIR = "artifacts"
INDEX_PATH = os.path.join(ART_DIR, "index.json")
IDF_PATH = os.path.join(ART_DIR, "idf.json")
FEATURES_PATH = os.path.join(ART_DIR, "selected_features.json")

def cmd_build(args):
    os.makedirs(ART_DIR, exist_ok=True)
    docs = load_txt_dir(args.data_dir)
    doc_tokens = {doc_id: preprocess(text) for doc_id, text in docs.items()}

    selected_terms, fs_report = select_features_df(doc_tokens, min_df=2, max_df_ratio=0.85, top_n=8000)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump({"selected_terms": sorted(selected_terms), "report": fs_report}, f, ensure_ascii=False)

    index = build_inverted_index(doc_tokens, allowed_terms=selected_terms)
    save_index(index, INDEX_PATH)

    idf = compute_idf(index, smooth=True)
    with open(IDF_PATH, "w", encoding="utf-8") as f:
        json.dump(idf, f, ensure_ascii=False)

    print(f"Built index for N={len(docs)} documents")
    print(f"Selected features: {fs_report.get('selected')} (from vocab {fs_report.get('vocab')})")
    print(f"Saved: {INDEX_PATH}, {IDF_PATH}, {FEATURES_PATH}")

def cmd_query(args):
    with open(IDF_PATH, "r", encoding="utf-8") as f:
        idf = {k: float(v) for k, v in json.load(f).items()}
    index = load_index(INDEX_PATH)
    model = TfidfModel(idf=idf)
    doc_vecs = build_doc_vectors(index, model)

    q_terms = preprocess(args.query)
    q_vec = build_query_vector(q_terms, model)
    ranked = retrieve(q_vec, index, doc_vecs, top_k=args.top_k)

    print(f"Query: {args.query}")
    for i, (doc_id, score) in enumerate(ranked, 1):
        print(f"{i:2d}) {doc_id}  score={score:.4f}")
        if args.summary:
            # summary needs raw text; load doc text (txt only for CLI)
            path = os.path.join(args.data_dir, doc_id)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                summ = summarize_extractive(text, idf=idf, num_sentences=2, max_chars=280)
                print(f"    summary: {summ}")
            except Exception:
                pass

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--data-dir", required=True)
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query")
    q.add_argument("--data-dir", required=True)
    q.add_argument("--query", required=True)
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument("--summary", action="store_true")
    q.set_defaults(func=cmd_query)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
