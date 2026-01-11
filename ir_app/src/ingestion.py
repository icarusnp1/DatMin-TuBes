import os

def noise_ratio(text):
    bad = sum(1 for c in text if ord(c) > 127)
    return bad / max(len(text), 1)

TXT_DIR = "data/raw_txt"

for fn in sorted(os.listdir(TXT_DIR)):
    if not fn.lower().endswith(".txt"):
        continue
    path = os.path.join(TXT_DIR, fn)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    length = len(text)
    ratio = noise_ratio(text)
    preview = " ".join(text.split()[:30])  # 30 kata pertama

    print("="*80)
    print(fn)
    print(f"chars   : {length}")
    print(f"noise   : {ratio:.4f}")
    print(f"preview : {preview}")
