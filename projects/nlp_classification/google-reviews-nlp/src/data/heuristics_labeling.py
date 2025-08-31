from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


RATING_PAT = re.compile(r"(\d+(?:[.,]\d+)?)\s*/\s*5")  # matches 4,7/5 | 4.7/5 | 3/5


def parse_rating_cell(raw: str | float | int | None) -> Optional[float]:
    """Try to parse rating from 'rating' column. Accepts '4.7/5', '4,7/5', '4.7', '3'..."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:/\s*5)?", s)
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    # Guard against 0-100 scales accidentally
    if val > 5.0:
        if val <= 100:
            val = val / 20.0  # rough clamp (100 -> 5.0)
        else:
            val = 5.0
    if val < 0:
        val = 0.0
    return val


def find_ratings_in_message(msg: str) -> List[float]:
    """Find all x/5 patterns inside message and return as floats."""
    if not isinstance(msg, str) or not msg:
        return []
    vals = []
    for m in RATING_PAT.finditer(msg):
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


def clean_message(text: str) -> str:
    """Light cleanup for message column; keeps the main content."""
    if pd.isna(text):
        return ""
    s = str(text)

    # Normalize slashes “ / ” to a single spaced separator
    s = re.sub(r"\s*/\s*", " / ", s)

    # Drop trailing “Comida/Serviço/Ambiente” blocks (optional)
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


def robust_read_csv(path: str | Path) -> pd.DataFrame:
    """
    Read pipe-delimited file with potential UTF-8 BOM, quotes, and long lines.
    Expect header: name|rating|number_of_photos|message
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {path.resolve()}\n"
            "Tip: run from the repo root (google-reviews-nlp) or pass --in with an absolute path.\n"
            "Expected structure: data/raw/clean-data-google-reviews.csv"
        )
    # Use engine='python' for safety; encoding 'utf-8-sig' to drop BOM if present
    df = pd.read_csv(
        path,
        sep="|",
        engine="python",
        encoding="utf-8-sig",
        dtype=str,           # keep raw forms; parse later
        on_bad_lines="skip", # skip malformed lines instead of breaking
    )
    # Ensure expected columns
    for col in ["name", "rating", "number_of_photos", "message"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: '{col}'. Found: {list(df.columns)}")
    return df


def main(input_csv: str, output_csv: str):
    df = robust_read_csv(input_csv)

    # Clean message
    df["message"] = df["message"].apply(clean_message)

    # Parse rating from 'rating' column
    base_rating = df["rating"].apply(parse_rating_cell)

    # Fallback: if rating is NaN/None, try to extract from message (e.g., 'Comida: 5/5 / ...')
    fallback_vals = []
    for msg, rv in zip(df["message"], base_rating):
        if rv is not None:
            fallback_vals.append(rv)
            continue
        candidates = find_ratings_in_message(msg)
        if not candidates:
            fallback_vals.append(None)
        else:
            # If multiple (Comida/Serviço/Ambiente), take the median as a robust summary
            fallback_vals.append(float(pd.Series(candidates).median()))
    df["rating_num"] = fallback_vals

    # To label
    df["label"] = df["rating_num"].apply(rating_to_label)

    # (Optional) drop rows with empty message or missing label
    df["message_len"] = df["message"].str.len().fillna(0).astype(int)
    out = df[(df["message_len"] > 0) & df["label"].notna()].copy()
    out = out.drop(columns=["message_len"])

    # Save
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(
        f"Saved labeled file to: {out_path.resolve()}\n"
        f"Total rows in: {len(df)} | usable rows out: {len(out)} | "
        f"label distribution:\n{out['label'].value_counts(dropna=False)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csv", default="data/raw/clean-data-google-reviews.csv")
    parser.add_argument("--out", dest="output_csv", default="data/processed/reviews_labeled.csv")
    args = parser.parse_args()
    main(args.input_csv, args.output_csv)