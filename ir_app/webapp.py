from __future__ import annotations

import os
import json
import re
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.loaders import load_corpus
from src.preprocessing import preprocess, preprocess_breakdown
from src.feature_selection import select_features_df
from src.indexing import (
    build_inverted_index,
    compute_idf,
    save_index,
    load_index,
    build_tf_df_idf_matrix,
)
from src.retrieval import (
    TfidfModel,
    build_doc_vectors,
    build_query_vector,
    retrieve,
    build_tfidf_matrix,
)
from src.summarization import summarize_extractive

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TXT_DIR = os.path.join(DATA_DIR, "raw_txt")
PDF_DIR = os.path.join(DATA_DIR, "raw_pdf")
INGESTED_TXT_DIR = os.path.join(DATA_DIR, "ingested_from_pdf")
ART_DIR = os.path.join(BASE_DIR, "artifacts")

INDEX_PATH = os.path.join(ART_DIR, "index.json")
IDF_PATH = os.path.join(ART_DIR, "idf.json")
FEATURES_PATH = os.path.join(ART_DIR, "selected_features.json")

SELECT_MIN_DF = 2
SELECT_MAX_DF_RATIO = 0.85
SELECT_TOP_N = 8000

os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(INGESTED_TXT_DIR, exist_ok=True)

app = FastAPI(title="Herbal IR")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

DOCS: Dict[str, str] = {}
INDEX: Optional[dict] = None
IDF: Dict[str, float] = {}
MODEL: Optional[TfidfModel] = None
DOC_VECS: Optional[dict] = None
FS_REPORT: Optional[dict] = None


def _load_artifacts_if_exist() -> bool:
    global DOCS, INDEX, IDF, MODEL, DOC_VECS, FS_REPORT

    if not os.path.exists(INDEX_PATH) or not os.path.exists(IDF_PATH):
        return False

    DOCS = load_corpus(txt_dir=TXT_DIR, pdf_dir=PDF_DIR, ingested_txt_dir=INGESTED_TXT_DIR)
    INDEX = load_index(INDEX_PATH)

    with open(IDF_PATH, "r", encoding="utf-8") as f:
        IDF = {k: float(v) for k, v in json.load(f).items()}

    MODEL = TfidfModel(idf=IDF)
    DOC_VECS = build_doc_vectors(INDEX, MODEL)

    if os.path.exists(FEATURES_PATH):
        try:
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                FS_REPORT = json.load(f).get("report")
        except Exception:
            FS_REPORT = None

    return True


def _rebuild_index() -> None:
    global DOCS, INDEX, IDF, MODEL, DOC_VECS, FS_REPORT

    DOCS = load_corpus(txt_dir=TXT_DIR, pdf_dir=PDF_DIR, ingested_txt_dir=INGESTED_TXT_DIR)
    doc_tokens = {doc_id: preprocess(text) for doc_id, text in DOCS.items()}

    selected_terms, FS_REPORT = select_features_df(
        doc_tokens,
        min_df=SELECT_MIN_DF,
        max_df_ratio=SELECT_MAX_DF_RATIO,
        top_n=SELECT_TOP_N,
    )
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump({"selected_terms": sorted(selected_terms), "report": FS_REPORT}, f, ensure_ascii=False)

    INDEX = build_inverted_index(doc_tokens, allowed_terms=selected_terms)
    save_index(INDEX, INDEX_PATH)

    IDF = compute_idf(INDEX, smooth=True)
    with open(IDF_PATH, "w", encoding="utf-8") as f:
        json.dump(IDF, f, ensure_ascii=False)

    MODEL = TfidfModel(idf=IDF)
    DOC_VECS = build_doc_vectors(INDEX, MODEL)


def _make_snippet(text: str, q_terms_raw: List[str], window: int = 260) -> str:
    if not text:
        return ""
    text_one = re.sub(r"\s+", " ", text).strip()
    low = text_one.lower()

    positions = []
    for t in set(q_terms_raw):
        pos = low.find(t.lower())
        if pos != -1:
            positions.append(pos)

    if not positions:
        return text_one[:window] + ("..." if len(text_one) > window else "")

    start = max(0, min(positions) - window // 3)
    end = min(len(text_one), start + window)
    snip = text_one[start:end]
    if start > 0:
        snip = "..." + snip
    if end < len(text_one):
        snip = snip + "..."
    return snip


def _highlight(text: str, terms: List[str]) -> str:
    out = text
    for t in sorted(set(terms), key=len, reverse=True):
        if len(t) < 3:
            continue
        out = re.sub(re.escape(t), lambda m: f"<mark>{m.group(0)}</mark>", out, flags=re.IGNORECASE)
    return out


def _tf_map(tokens: List[str]) -> Dict[str, int]:
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


@app.on_event("startup")
def startup():
    _load_artifacts_if_exist()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    has_index = os.path.exists(INDEX_PATH) and os.path.exists(IDF_PATH)
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "has_index": has_index, "doc_count": len(DOCS) if has_index else 0},
    )


@app.post("/search", response_class=HTMLResponse)
def search(request: Request, q: str = Form(...), top_k: int = Form(10)):
    global INDEX, MODEL, DOC_VECS

    if INDEX is None or MODEL is None or DOC_VECS is None:
        if not _load_artifacts_if_exist():
            return templates.TemplateResponse(
                "home.html",
                {
                    "request": request,
                    "has_index": False,
                    "doc_count": 0,
                    "error": "Index belum dibangun. Buka Admin dan klik Rebuild Index.",
                },
            )

    q_terms_raw = re.findall(r"[a-zA-Z]+", (q or "").lower())

    # preprocessing query (hasil final untuk retrieval)
    q_terms = preprocess(q)

    # breakdown preprocessing query untuk ditampilkan
    query_prep = preprocess_breakdown(q)

    q_vec = build_query_vector(q_terms, MODEL)
    ranked = retrieve(q_vec, INDEX, DOC_VECS, top_k=top_k)

    results = []
    doc_prep = []
    for doc_id, score in ranked:
        text = DOCS.get(doc_id, "")
        snippet = _make_snippet(text, q_terms_raw)
        summary = summarize_extractive(text, idf=MODEL.idf, num_sentences=2, max_chars=320)
        results.append(
            {
                "doc_id": doc_id,
                "score": float(score),
                "snippet": _highlight(snippet, q_terms_raw),
                "summary": _highlight(summary, q_terms_raw),
            }
        )

        if len(doc_prep) < 5:
            doc_prep.append({"doc_id": doc_id, "prep": preprocess_breakdown(text)})

    # Matrix: TF + DF + IDF (+ kolom Q) untuk term query
    q_tf = _tf_map(q_terms)
    terms_used = sorted(q_tf.keys())
    doc_ids_in_table = [doc_id for doc_id, _ in ranked]

    tf_df_idf_matrix = build_tf_df_idf_matrix(
        index=INDEX,
        idf=MODEL.idf,
        doc_ids=doc_ids_in_table,
        terms=terms_used,
        q_tf=q_tf,  # tampilkan kolom Q
    )

    # Matrix baru: TF-IDF (RAW) (+ kolom Q) untuk term query
    tfidf_matrix = build_tfidf_matrix(
        index=INDEX,
        model=MODEL,
        doc_ids=doc_ids_in_table,
        terms=terms_used,
        q_tf=q_tf,
    )

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "q": q,
            "results": results,
            "doc_count": len(DOCS),
            "preprocessing_view": {"query": query_prep, "docs": doc_prep},
            "tf_df_idf_matrix": tf_df_idf_matrix,
            "tfidf_matrix": tfidf_matrix,
        },
    )


@app.get("/doc/{doc_id}", response_class=HTMLResponse)
def doc_detail(request: Request, doc_id: str, q: str = ""):
    text = DOCS.get(doc_id)
    if text is None:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Dokumen tidak ditemukan."})

    q_terms_raw = re.findall(r"[a-zA-Z]+", q.lower()) if q else []

    excerpt = re.sub(r"\s+", " ", text).strip()
    excerpt = excerpt[:4000] + ("..." if len(excerpt) > 4000 else "")
    excerpt = _highlight(excerpt, q_terms_raw)

    summary = summarize_extractive(text, idf=IDF, num_sentences=3, max_chars=450) if IDF else ""
    summary = _highlight(summary, q_terms_raw)

    return templates.TemplateResponse(
        "doc.html",
        {"request": request, "doc_id": doc_id, "q": q, "summary": summary, "content": excerpt},
    )


@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    global INDEX, MODEL, DOC_VECS

    has_index = os.path.exists(INDEX_PATH) and os.path.exists(IDF_PATH)

    fs = None
    if os.path.exists(FEATURES_PATH):
        try:
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                fs = json.load(f).get("report")
        except Exception:
            fs = None

    ingest_previews = []
    try:
        if os.path.isdir(INGESTED_TXT_DIR):
            for fn in sorted(os.listdir(INGESTED_TXT_DIR)):
                if not fn.lower().endswith(".txt"):
                    continue
                text = DOCS.get(fn, "")
                ingest_previews.append({"doc_id": fn, "preview": text[:800] + ("..." if len(text) > 800 else "")})
                if len(ingest_previews) >= 5:
                    break
    except Exception:
        ingest_previews = []

    prep_samples = []
    try:
        for doc_id in list(DOCS.keys())[:5]:
            prep_samples.append({"doc_id": doc_id, "prep": preprocess_breakdown(DOCS.get(doc_id, ""))})
    except Exception:
        prep_samples = []

    admin_tf_df_idf_matrix = None
    admin_tfidf_matrix = None
    if has_index:
        if INDEX is None or MODEL is None or DOC_VECS is None or not IDF:
            _load_artifacts_if_exist()

        if INDEX is not None and MODEL is not None and DOCS:
            doc_ids_all = sorted(DOCS.keys())

            # SEMUA term × SEMUA dokumen: TF + DF + IDF
            admin_tf_df_idf_matrix = build_tf_df_idf_matrix(
                index=INDEX,
                idf=IDF,
                doc_ids=doc_ids_all,
                terms=None,   # semua term
                q_tf=None,    # tanpa query
            )

            # SEMUA term × SEMUA dokumen: TF-IDF (RAW)
            admin_tfidf_matrix = build_tfidf_matrix(
                index=INDEX,
                model=MODEL,
                doc_ids=doc_ids_all,
                terms=None,
                q_tf=None,
            )

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "has_index": has_index,
            "doc_count": len(DOCS) if has_index else 0,
            "fs": fs,
            "ingest_previews": ingest_previews,
            "prep_samples": prep_samples,
            "admin_tf_df_idf_matrix": admin_tf_df_idf_matrix,
            "admin_tfidf_matrix": admin_tfidf_matrix,
        },
    )


@app.post("/admin/rebuild")
def admin_rebuild():
    _rebuild_index()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...)):
    name = file.filename or "upload"
    ext = os.path.splitext(name)[1].lower()

    if ext == ".txt":
        out = os.path.join(TXT_DIR, os.path.basename(name))
        data = await file.read()
        with open(out, "wb") as f:
            f.write(data)
    elif ext == ".pdf":
        out = os.path.join(PDF_DIR, os.path.basename(name))
        data = await file.read()
        with open(out, "wb") as f:
            f.write(data)
    else:
        return RedirectResponse(
            url="/admin?err=Format%20tidak%20didukung.%20Gunakan%20TXT%20atau%20PDF.",
            status_code=303,
        )

    return RedirectResponse(url="/admin", status_code=303)