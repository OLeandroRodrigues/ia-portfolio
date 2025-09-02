# tests/test_heuristics_labeling.py
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

# Import the module by its dotted path exactly from your tree
# Adjust this string if your module path differs
pp = importlib.import_module("src.data.heuristics_labeling")



# ---------------------------------
# Helpers
# ---------------------------------
def _write_pipe_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    """Write a pipe-delimited CSV with UTF-8 encoding and Unix newlines."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("|".join(header) + "\n")
        for r in rows:
            f.write("|".join("" if v is None else str(v) for v in r) + "\n")

def _first_row_by_name(df: pd.DataFrame, name: str) -> pd.Series:
        """Return the first row where df['name'] == name, or fail with a helpful message."""
        subset = df[df["name"].astype(str).eq(name)]
        assert not subset.empty, (
            f"Expected a row for name == {name!r}, but none was found. "
            f"Present names: {sorted(df['name'].astype(str).unique().tolist())}"
        )
        return subset.iloc[0]


class TestHeuristicsLabeling:
    # ----------------------------
    # Unit tests: parse_rating_cell
    # ----------------------------
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("4.7/5", 4.7),
            ("4,7/5", 4.7),
            ("3/5", 3.0),
            ("5/5", 5.0),
            ("0/5", 0.0),
            ("4.7", 4.7),
            ("3", 3.0),
            (None, None),
            ("", None),
            (float("nan"), None),
            # Values above 5: 20 -> 1.0, 100 -> 5.0, 87 -> 4.35
            ("20", 1.0),
            ("100", 5.0),
            ("87", 87 / 20.0),
            # Negative values are clamped to 0.0
            ("-3", 0.0),
            # Very large values are clamped to 5.0
            ("1000", 5.0),
        ],
    )
    def test_parse_rating_cell(self, raw, expected):
        out = pp.parse_rating_cell(raw)
        if expected is None:
            assert out is None
        else:
            assert pytest.approx(out, rel=1e-9) == expected

    # ----------------------------
    # Unit tests: find_ratings_in_message
    # ----------------------------
    @pytest.mark.parametrize(
        "msg,expected",
        [
            ("Comida: 4,5/5 | Serviço: 3/5 | Ambiente: 5/5", [4.5, 3.0, 5.0]),
            ("nota 4.7/5 e depois 4/5", [4.7, 4.0]),
            ("sem notas", []),
            ("limites 5/5 e 0/5 são válidos", [5.0, 0.0]),
            ("fora dos limites 6/5 deve ignorar", []),
            (None, []),
            ("", []),
        ],
    )
    def test_find_ratings_in_message(self, msg, expected):
        assert pp.find_ratings_in_message(msg) == expected

    # ----------------------------
    # Unit tests: rating_to_label
    # ----------------------------
    @pytest.mark.parametrize(
        "val,expected",
        [
            (None, None),
            (4.0, "positive"),
            (5.0, "positive"),
            (3.9999, "neutral"),
            (2.0000, "negative"),
            (1.5, "negative"),
            (2.0001, "neutral"),
            (3.0, "neutral"),
        ],
    )
    def test_rating_to_label(self, val, expected):
        assert pp.rating_to_label(val) == expected

    # ----------------------------
    # Unit tests: clean_message
    # ----------------------------
    def test_clean_message_basic_cleanup(self):
        text = "  oi / tudo  bem?  "
        # Normalize " / " and collapse extra spaces
        assert pp.clean_message(text) == "oi / tudo bem?"

    def test_clean_message_remove_urls_and_collapse_space(self):
        text = "Veja http://example.com agora!\nOutro  link: https://foo.bar/x"
        out = pp.clean_message(text)
        # URLs must be removed and whitespace collapsed
        assert "http" not in out
        assert "  " not in out
        assert out.startswith("Veja") and out.endswith("Outro link:")

    def test_clean_message_drop_comida_servico_ambiente_block(self):
        text = "Ótimo lugar! Comida: 4,5/5 / Serviço: 4/5 / Ambiente: 5/5"
        out = pp.clean_message(text)
        # The trailing "Comida/Serviço/Ambiente" block must be removed
        assert out == "Ótimo lugar!"

    # ----------------------------
    # Unit tests: robust_read_csv
    # ----------------------------
    def test_robust_read_csv_success(self, tmp_path: Path):
        p = tmp_path / "in.csv"
        header = ["name", "rating", "number_of_photos", "message"]
        rows = [
            ["ana", "4.7/5", "3", "muito bom"],
            ["carlos", "3/5", "0", "ok"],
        ]
        _write_pipe_csv(p, header, rows)

        df = pp.robust_read_csv(p)
        assert list(df.columns) == header
        assert len(df) == 2
        # dtype=str in read_csv ensures everything is a string
        assert df.loc[0, "rating"] == "4.7/5"

    def test_robust_read_csv_missing_file(self, tmp_path: Path):
        missing = tmp_path / "nope.csv"
        with pytest.raises(FileNotFoundError) as ex:
            pp.robust_read_csv(missing)
        assert "Input CSV not found" in str(ex.value)

    def test_robust_read_csv_missing_column(self, tmp_path: Path):
        p = tmp_path / "bad.csv"
        header = ["name", "rating", "message"]  # missing number_of_photos
        rows = [["ana", "4.7/5", "texto"]]
        _write_pipe_csv(p, header, rows)
        with pytest.raises(ValueError) as ex:
            pp.robust_read_csv(p)
        assert "Missing expected column" in str(ex.value)
    
    # ----------------------------
    # Integration test: main()
    # ----------------------------
    def test_main_end_to_end(self, tmp_path: Path):
        """Validate the complete pipeline:
        - Reads a pipe-delimited CSV with string values
        - Cleans the message (removes Comida/Serviço/Ambiente block and URLs)
        - Extracts rating from 'rating' column, or falls back to ratings in message
        - Applies sentiment labeling
        - Drops rows with empty message or missing label
        - Writes the final CSV to disk
        """
        input_csv = tmp_path / "raw.csv"
        output_csv = tmp_path / "out.csv"

        header = ["name", "rating", "number_of_photos", "message"]
        rows = [
            # 1) rating present in the column
            ["ana", "4.7/5", "2", "Ótimo! http://x.y"],
            # 2) rating missing in column; keep some text BEFORE the block so the message
            #    is not empty after cleaning. (Fallback median of 3,4,5 = 4.0)
            ["bia", "", "0", "Avaliação geral. Comida: 3/5 / Serviço: 5/5 / Ambiente: 4/5"],
            # 3) no rating at all => should be dropped
            ["caio", "", "1", "sem nota aqui"],
            # 4) empty message => should be dropped
            ["duda", "5/5", "0", ""],
            # 5) scale 100 => converted to 5.0 => positive
            ["enzo", "100", "0", "texto qualquer"],
            # 6) negative rating => clamped to 0.0 => negative
            ["fabi", "-3", "0", "ruim"],
        ]
        _write_pipe_csv(input_csv, header, rows)

        # Run the full pipeline
        pp.main(str(input_csv), str(output_csv))

        # Check output file exists
        assert output_csv.exists(), "Output file was not created"

        out = pd.read_csv(output_csv)

        # Expected columns
        assert {"name", "rating", "number_of_photos", "message", "rating_num", "label"}.issubset(out.columns)

        # 1) ana => rating 4.7 -> positive
        row_ana = _first_row_by_name(out, "ana")
        assert row_ana["label"] == "positive"
        assert "http" not in row_ana["message"]

        # 2) enzo => 100 -> 5.0 -> positive
        row_enzo = _first_row_by_name(out, "enzo")
        assert pytest.approx(row_enzo["rating_num"]) == 5.0
        assert row_enzo["label"] == "positive"

        # 3) fabi => -3 -> 0.0 -> negative
        row_fabi = _first_row_by_name(out, "fabi")
        assert pytest.approx(row_fabi["rating_num"]) == 0.0
        assert row_fabi["label"] == "negative"
