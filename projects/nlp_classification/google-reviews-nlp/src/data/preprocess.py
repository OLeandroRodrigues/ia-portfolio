from __future__ import annotations
import re
import argparse
import pandas as pd
from pathlib import Path


COLUMNS_NEEDED = ["name","rating","number_of_photos","message"]  # columns in the .csv file

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = re.sub(r"http[s]?://\S+", " ", t)     # remove URLs
    t = re.sub(r"\s+", " ", t)                # normalize spaces
    return t.lower().strip()

def prepare_columns(df: pd.DataFrame, name_field: str, default_if_empty: str = "") -> pd.DataFrame:
    
    if name_field not in df.columns:
        raise KeyError(f"Missing required column: '{name_field}'")

    # Avoiding "nan" string: use dtype string native and just after strip
    s = df[name_field].astype("string").fillna("").str.strip()

    # specific rule: empty number_of_photos field -> "0" (or default)
    if name_field == "number_of_photos":
        # replaces only literal empty values
        s = s.mask(s.eq(""), default_if_empty)

    s_clean = s.apply(basic_clean).astype("string")

    df[f"{name_field}_clean"] = s_clean
    df[f"{name_field}_length"] = s_clean.str.len()

    return df


def run(input_csv: str, output_csv: str) -> None:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    # More resilient to bad rows
    df = pd.read_csv(
        input_csv,
        sep="|",
        engine="python",
        on_bad_lines="skip",  # pandas >= 1.3
    )

    # (Opcional) validation minimum columns 
    missing = [c for c in ["name", "rating", "number_of_photos", "message"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # pipeline
    df = prepare_columns(df, "name")
    df = prepare_columns(df, "rating")
    df = prepare_columns(df, "number_of_photos", default_if_empty="0")
    df = prepare_columns(df, "message")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[preprocess] saved -> {output_csv} (rows={len(df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/data-google-reviews.csv")
    ap.add_argument("--output", default="data/processed/data-google-reviews-clean.csv")
    args = ap.parse_args()
    run(args.input, args.output)