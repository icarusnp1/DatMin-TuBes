from __future__ import annotations
import os, json, re
from typing import Dict, List
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.loaders import load_corpus
from src.preprocessing import preprocess
from src.indexing import build_inverted_index, compute_idf, save_index, load_index
from src.retrieval import TfidfModel, build_doc_vectors, build_query_vector, retrieve
from src.summarization import summarize_extractive

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TXT_DIR = os.path.join(DATA_DIR, "raw_txt")
PDF_DIR = os.path.join(DATA_DIR, "raw_pdf")
INGESTED_TXT_DIR = os.path.join(DATA_DIR, "ingested_from_pdf")
ART_DIR = os.path.join(BASE_DIR, "artifacts")
INDEX_PATH = os.path.join(ART_DIR, "index.json")
IDF_PATH = os.path.join(ART_DIR, "idf.json")

os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(INGESTED_TXT_DIR, exist_ok=True)

app = FastAPI(title="Herbal IR")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

DOCS: Dict[str, str] = {}
INDEX = None
IDF: Dict[str, float] = {}
MODEL = None
DOC_VECS = None

def _load_artifacts_if_exist() -> bool:
    global DOCS, INDEX, IDF, MODEL, DOC_VECS
    if not os.path.exists(INDEX_PATH) or not os.path.exists(IDF_PATH):
        return False
    DOCS = load_corpus(txt_dir=TXT_DIR, pdf_dir=PDF_DIR, ingested_txt_dir=INGESTED_TXT_DIR)
    INDEX = load_index(INDEX_PATH)
    with open(IDF_PATH, "r", encoding="utf-8") as f:
        IDF = {k: float(v) for k, v in json.load(f).items()}
    MODEL = TfidfModel(idf=IDF)
    DOC_VECS = build_doc_vectors(INDEX, MODEL)
    return True

def _rebuild_index() -> None:
    global DOCS, INDEX, IDF, MODEL, DOC_VECS
    DOCS = load_corpus(txt_dir=TXT_DIR, pdf_dir=PDF_DIR, ingested_txt_dir=INGESTED_TXT_DIR)
    doc_tokens = {doc_id: preprocess(text) for doc_id, text in DOCS.items()}
    INDEX = build_inverted_index(doc_tokens)
    save_index(INDEX, INDEX_PATH)
    IDF = compute_idf(INDEX, smooth=True)
    with open(IDF_PATH, "w", encoding="utf-8") as f:
        json.dump(IDF, f, ensure_ascii=False)
    MODEL = TfidfModel(idf=IDF)
    DOC_VECS = build_doc_vectors(INDEX, MODEL)

def _make_snippet(text: str, q_terms_raw: List[str], window: int = 260) -> str:
    if not text:
        return ""
    low = text.lower()
    positions = []
    for t in set(q_terms_raw):
        pos = low.find(t.lower())
        if pos != -1:
            positions.append(pos)
    text_one = re.sub(r"\s+", " ", text).strip()
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

@app.on_event("startup")
def startup():
    _load_artifacts_if_exist()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    has_index = os.path.exists(INDEX_PATH) and os.path.exists(IDF_PATH)
    return templates.TemplateResponse("home.html", {"request": request, "has_index": has_index, "doc_count": len(DOCS) if has_index else 0})

@app.post("/search", response_class=HTMLResponse)
def search(request: Request, q: str = Form(...), top_k: int = Form(10)):
    global INDEX, MODEL, DOC_VECS
    if INDEX is None or MODEL is None or DOC_VECS is None:
        if not _load_artifacts_if_exist():
            return templates.TemplateResponse("home.html", {"request": request, "has_index": False, "doc_count": 0, "error": "Index belum dibangun. Buka Admin dan klik Rebuild Index."})

    q_terms_raw = re.findall(r"[a-zA-Z]+", q.lower())
    q_terms = preprocess(q)
    q_vec = build_query_vector(q_terms, MODEL)
    ranked = retrieve(q_vec, INDEX, DOC_VECS, top_k=top_k)

    results = []
    for doc_id, score in ranked:
        text = DOCS.get(doc_id, "")
        snippet = _make_snippet(text, q_terms_raw)
        summary = summarize_extractive(text, idf=MODEL.idf, num_sentences=2, max_chars=320)
        results.append({
            "doc_id": doc_id,
            "score": float(score),
            "snippet": _highlight(snippet, q_terms_raw),
            "summary": _highlight(summary, q_terms_raw),
        })

    return templates.TemplateResponse("results.html", {"request": request, "q": q, "results": results, "doc_count": len(DOCS)})

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
    return templates.TemplateResponse("doc.html", {"request": request, "doc_id": doc_id, "q": q, "summary": summary, "content": excerpt})

@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    has_index = os.path.exists(INDEX_PATH) and os.path.exists(IDF_PATH)
    return templates.TemplateResponse("admin.html", {"request": request, "has_index": has_index, "doc_count": len(DOCS) if has_index else 0})

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
        return RedirectResponse(url="/admin?err=Format%20tidak%20didukung.%20Gunakan%20TXT%20atau%20PDF.", status_code=303)
    return RedirectResponse(url="/admin", status_code=303)
