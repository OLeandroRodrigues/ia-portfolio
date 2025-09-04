# src/data/heuristics_labeling.py
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Reconfigure stdout/stderr to UTF-8 (helps on Windows consoles)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Regex to remove emojis and various pictographic symbols
# Covers: emoticons, pictographs, transport, flags, dingbats, variation selectors, skin tones, etc.
EMOJI_PAT = re.compile(
    "["                     # start char class
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric ext
    "\U0001F800-\U0001F8FF"  # supplemental arrows
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess, symbols
    "\U0001FA70-\U0001FAFF"  # emoji ext-A
    "\U00002700-\U000027BF"  # dingbats (includes ❤ etc.)
    "\U00002600-\U000026FF"  # misc symbols (☕ etc.)
    "\U00002B00-\U00002BFF"  # arrows etc.
    "\U0000FE0F"             # variation selector-16
    "\U0001F1E6-\U0001F1FF"  # flags
    "]",
    flags=re.UNICODE,
)

RATING_IN_MSG_PAT = re.compile(r"(\d+(?:[.,]\d+)?)\s*/\s*5", re.IGNORECASE)

# ---------- rating helpers ----------

def parse_rating_cell(raw: str | float | int | None) -> Optional[float]:
    """Parse rating from 'rating' column. Accepts '4.7/5', '4,7/5', '4.7', '3'."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:/\s*5)?", s)
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    # guard for accidental 0-100 scales
    if val > 5.0:
        if val <= 100:
            val = val / 20.0
        else:
            val = 5.0
    if val < 0:
        val = 0.0
    return val


def find_ratings_in_message(msg: str) -> List[float]:
    """Find all x/5 patterns in the message text and return as floats."""
    if not isinstance(msg, str) or not msg:
        return []
    vals = []
    for m in RATING_IN_MSG_PAT.finditer(msg):
        v = float(m.group(1).replace(",", "."))
        if 0.0 <= v <= 5.0:
            vals.append(v)
    return vals


def rating_to_label(v: Optional[float]) -> Optional[str]:
    """Map numeric rating to sentiment label."""
    if v is None:
        return None
    if v >= 4.0:
        return "positive"
    if v <= 2.0:
        return "negative"
    return "neutral"


# ---------- cleaning ----------

def remove_emojis(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return EMOJI_PAT.sub("", text)


def clean_message(text: str, strip_emojis: bool = True) -> str:
    """Light cleanup for message."""
    if pd.isna(text):
        return ""
    s = str(text)

    # Optionally remove emojis and pictographs
    if strip_emojis:
        s = remove_emojis(s)

    # Normalize slashes and common UI artifacts
    s = re.sub(r"\s*/\s*", " / ", s)

    # Drop trailing 'Comida/Serviço/Ambiente' rating blocks (optional)
    s = re.sub(
        r"(Comida:\s*\d+(?:[.,]\d+)?\s*/\s*5.*)$",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # Remove URLs
    s = re.sub(r"http\S+", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- IO ----------

def robust_read_csv(path: str | Path) -> pd.DataFrame:
    """
    Read pipe-delimited file with possible BOM/quotes/long lines.
    Expect header: name|rating|number_of_photos|message
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {path.resolve()}\n"
            "Tip: run from the repo root (google-reviews-nlp) or pass --in with an absolute path.\n"
            "Expected structure: data/raw/data-google-reviews.csv"
        )
    df = pd.read_csv(
        path,
        sep="|",
        engine="python",
        encoding="utf-8-sig",
        dtype=str,            # keep as string; parse later
        on_bad_lines="skip",
    )
    for col in ["name", "rating", "number_of_photos", "message"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: '{col}'. Found: {list(df.columns)}")
    return df


# ---------- main ----------

def main(input_csv: str, output_csv: str, strip_emojis: bool = True):
    df = robust_read_csv(input_csv)

    # Clean message (with emoji removal by default)
    df["message"] = df["message"].apply(lambda x: clean_message(x, strip_emojis=strip_emojis))

    # Parse rating from column, or fallback to message patterns
    base_rating = df["rating"].apply(parse_rating_cell)

    final_vals: List[Optional[float]] = []
    for msg, rv in zip(df["message"], base_rating):
        if rv is not None:
            final_vals.append(rv)
            continue
        candidates = find_ratings_in_message(msg)
        if not candidates:
            final_vals.append(None)
        else:
            # median of multiple sub-ratings (Comida/Serviço/Ambiente)
            final_vals.append(float(pd.Series(candidates).median()))

    df["rating_num"] = final_vals
    df["label"] = df["rating_num"].apply(rating_to_label)

    # Drop rows without message or label
    df["message_len"] = df["message"].str.len().fillna(0).astype(int)
    out = df[(df["message_len"] > 0) & df["label"].notna()].copy()
    out = out.drop(columns=["message_len"])

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Print a small summary (safe for Windows due to reconfigure above)
    print(
        f"Saved labeled file to: {out_path.resolve()}\n"
        f"Total rows in: {len(df)} | usable rows out: {len(out)}\n"
        f"Label distribution:\n{out['label'].value_counts(dropna=False)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csv", default="data/raw/data-google-reviews.csv")
    parser.add_argument("--out", dest="output_csv", default="data/processed/reviews_labeled.csv")
    parser.add_argument("--keep-emojis", action="store_true", help="Do NOT strip emojis from messages")
    args = parser.parse_args()
    main(args.input_csv, args.output_csv, strip_emojis=(not args.keep_emojis))