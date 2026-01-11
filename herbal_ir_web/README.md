# Herbal IR (Website + Full Build)

## Features
- Preprocessing (includes Porter-ID stemming) + domain stopwords
- Inverted index
- TF-IDF + cosine (VSM)
- Snippet + extractive summarization (safe for PDF text)
- Website (FastAPI):
  - Search page (query -> ranked files)
  - Results page (score, snippet, ringkasan)
  - Document detail page
  - Admin page: upload TXT/PDF and rebuild index

## Install
```bash
pip install -r requirements.txt
```

## Run the website
From project root:
```bash
uvicorn webapp:app --reload --port 8000
```
Open:
- Home:  http://127.0.0.1:8000/
- Admin: http://127.0.0.1:8000/admin

## Data
- Put TXT into `data/raw_txt`
- Put PDF into `data/raw_pdf` (it will be converted into `data/ingested_from_pdf` during rebuild)
- Go to /admin and click Rebuild Index

## Swap stemmer
Replace only `src/stemmer_porter_id.py` (keep API: stem_word, stem_tokens).
