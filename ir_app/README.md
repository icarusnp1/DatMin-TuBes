# Herbal IR (Website + Feature Selection)

## What you get
- Preprocessing: tokenize + stopwords (general + herbal domain) + Porter-style Indonesian stemming (swappable)
- Feature selection: DF threshold + Top-N (unsupervised), saved to `artifacts/selected_features.json`
- Inverted index (built only on selected features)
- TF-IDF + cosine similarity (VSM)
- Snippet + extractive summarization (robust for PDF-extracted text)
- Website (FastAPI): Search, Results, Detail, Admin upload, Rebuild Index

## Install
```bash
pip install -r requirements.txt
```

## Run website
```bash
uvicorn webapp:app --reload --port 8000
```
Open:
- http://127.0.0.1:8000/
- Admin: http://127.0.0.1:8000/admin

## Data folders
- `data/raw_txt`: place .txt files
- `data/raw_pdf`: place .pdf files (will be converted to `data/ingested_from_pdf` during rebuild)

## CLI (optional)
```bash
python -m src.app build --data-dir data/raw_txt
python -m src.app query --data-dir data/raw_txt --query "temulawak sehat" --top-k 10 --summary
```

## Replace stemmer only
Replace `src/stemmer_porter_id.py` only, keep API: `stem_word`, `stem_tokens`.
