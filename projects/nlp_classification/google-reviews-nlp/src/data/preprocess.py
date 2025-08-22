from __future__ import annotations
import re
import argparse
import pandas as pd
from pathlib import Path


COLUMNS_NEEDED = ["name","rating","number_of_fotos","message"]  # columns in the .csv file

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = re.sub(r"http[s]?://\S+", " ", t)     # remove URLs
    t = re.sub(r"\s+", " ", t)                # normalize spaces
    return t.lower()

def prepare_columns(df: pd.DataFrame, name_field: str, default_if_empty: str = "") -> pd.DataFrame:
    
    # force everything to string to avoid errors in basic_clean
    s = df[name_field].astype(str).fillna("").str.strip()

    # specific rule: if the field is number_of_photos and comes empty, set it to '0'
    if name_field == "number_of_photos":
        s = s.replace("", default_if_empty)

    s_clean = s.apply(basic_clean) 
    df[f"{name_field}_clean"] = s_clean
    df[f"{name_field}_length"] = s_clean.str.len()

    return df


def run(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv, sep="|")

    # fields name
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
    ap.add_argument("--output", default="data/processed/data-google-reviews_clean.csv")
    args = ap.parse_args()
    run(args.input, args.output)