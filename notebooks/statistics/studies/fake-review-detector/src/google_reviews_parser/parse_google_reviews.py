# parse_reviews_pandas.py
# -*- coding: utf-8 -*-
import re
import pandas as pd
from pathlib import Path
import os

# Gets the folder where this .py file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Goes up two levels
project_root = os.path.dirname(os.path.dirname(current_dir))

# Path to the "data" folder (already existing)
data_dir = os.path.join(project_root, "data")

# Final file path inside the "data" folder
path = os.path.join(data_dir, "dirty-data-google-reviews.txt")

#INFILE = Path("input.txt")
INFILE = Path(path)
OUTFILE = Path(data_dir,"clean-data-google-reviews.txt")

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s*\|\s*", " | ", s)
    return s

def clean_message(s: str) -> str:
    noise_patterns = [
        r"Passe o cursor para reagir",
        r"Ofner\s*-\s*Perdizes\s*\(proprietário\).*?(?=(?:\n|$))",
    ]
    for pat in noise_patterns:
        s = re.sub(pat, "", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    return s

def extract_rating(text: str) -> str:
    vals = []
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)[ ]*/[ ]*5\b", text):
        num = m.group(1).replace(",", ".")
        try:
            vals.append(float(num))
        except ValueError:
            pass
    if vals:
        avg = sum(vals) / len(vals)
        return f"{avg:.1f}/5"
    return ""  # accept with no rating

def sanitize_field(s: str) -> str:
    return s.replace("|", " / ").strip()

# --------- Regex mais flexível ---------
# Aceita: nome + (opcional: Local Guide) + (opcional: X avaliações) + (opcional: Y fotos) + resto
HEADER_RE = re.compile(
    r"""
    (?P<name>[^|\n]+?)                 # name until pipe or breakline (not ambitious )
    \s+
    (?:Local\ Guide·\s*)?              # 'Local Guide·' (opcional)
    (?P<reviews>\d+)\s+avaliaç(?:ões|ão)   # show 'X reviews' ou '1 review'
    (?:·(?P<photos>\d+)\s+fotos)?      # '· Y photos' (opcional)
    """,
    re.IGNORECASE | re.VERBOSE
)
def find_headers(text: str):
    return list(HEADER_RE.finditer(text))

def parse_to_df(text: str) -> pd.DataFrame:
    text = normalize_text(text)
    headers = find_headers(text)
    rows = []

    for i, h in enumerate(headers):
        name = h.group("name").strip()
        photos = h.group("photos") or ""
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].lstrip(" |").strip()

        rating = extract_rating(body)
        body = clean_message(body)

        rows.append({
            "name": sanitize_field(name),
            "rating": sanitize_field(rating),
            "number_of_photos": sanitize_field(photos),
            "message": sanitize_field(body)
        })

    df = pd.DataFrame(rows, columns=["name", "rating", "number_of_photos", "message"])
    return df

def main():
    if not INFILE.exists():
        print(f"[ERROR] File {INFILE} not found.")
        return
    raw = INFILE.read_text(encoding="utf-8")
    df = parse_to_df(raw)
    df.to_csv(OUTFILE, sep="|", index=False, encoding="utf-8")
    print(f"[OK] File '{OUTFILE}' generated with {len(df)} records.")

if __name__ == "__main__":
    main()