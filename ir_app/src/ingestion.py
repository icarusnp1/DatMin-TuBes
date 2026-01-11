from pdfminer.high_level import extract_text
import os

PDF_DIR = "data/raw_pdf"
TXT_DIR = "data/raw_txt"

os.makedirs(TXT_DIR, exist_ok=True)

def pdf_to_text(pdf_path):
    return extract_text(pdf_path)

for fname in os.listdir(PDF_DIR):
    if not fname.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_DIR, fname)
    txt_path = os.path.join(
        TXT_DIR, fname.replace(".pdf", ".txt")
    )

    text = pdf_to_text(pdf_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[OK] {fname} → {len(text)} karakter")
