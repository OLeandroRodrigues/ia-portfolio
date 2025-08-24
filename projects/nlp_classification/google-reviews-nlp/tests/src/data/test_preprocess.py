# tests/test_preprocess.py
# Run with: python -m pytest -q

from pathlib import Path
import csv
import sys
import subprocess
import importlib
import types
import pandas as pd
import pytest

# Import the module exactly from your tree: src/data/preprocess.py
pp = importlib.import_module("src.data.preprocess")


class TestPreprocess:
    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _write_pipe_csv(path: Path, rows: list[dict]):
        """Write a pipe-delimited CSV (|) to match the reader in run()."""
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="|")
            writer.writeheader()
            writer.writerows(rows)

    # -----------------------------
    # basic_clean
    # -----------------------------
    def test_basic_clean_removes_urls_normalizes_spaces_and_lowercases(self):
        """basic_clean should drop URLs, collapse whitespace, and lowercase."""
        assert hasattr(pp, "basic_clean") and isinstance(pp.basic_clean, types.FunctionType)

        txt = "  Visit https://example.com  NOW!   \nAnd see http://foo.bar/x  "
        out = pp.basic_clean(txt)

        # URLs gone
        assert "http" not in out and "example.com" not in out and "foo.bar" not in out
        # Spaces normalized + lowercased
        assert out == "visit now! and see"

    def test_basic_clean_non_string_returns_empty(self):
        """Non-string inputs must return empty string."""
        assert pp.basic_clean(None) == ""
        assert pp.basic_clean(123) == ""

    # -----------------------------
    # prepare_columns
    # -----------------------------
    def test_prepare_columns_adds_clean_and_length_and_specific_rules(self):
        """
        prepare_columns should:
        - create <field>_clean and <field>_length
        - use string dtype to avoid literal 'nan'
        - for number_of_photos: empty -> default ('0')
        - apply basic_clean (remove URLs, trim, lowercase)
        """
        df = pd.DataFrame({
            "name": ["  Ana  ", "CARLOS", None],
            "rating": [" 5 ", "4", "  3"],
            "number_of_photos": ["", "10", None],
            "message": [" Hi  http://spam ", "   Tudo   bem ", None],
        })

        df = pp.prepare_columns(df, "name")
        df = pp.prepare_columns(df, "rating")
        df = pp.prepare_columns(df, "number_of_photos", default_if_empty="0")
        df = pp.prepare_columns(df, "message")

        # columns created
        for col in ["name", "rating", "number_of_photos", "message"]:
            assert f"{col}_clean" in df.columns
            assert f"{col}_length" in df.columns

        # cleaning behavior
        assert df.loc[0, "name_clean"] == "ana"
        assert df.loc[1, "name_clean"] == "carlos"
        assert df.loc[2, "name_clean"] == ""

        # numbers preserved as strings after trimming
        assert df["rating_clean"].tolist() == ["5", "4", "3"]

        # specific rule for number_of_photos empty -> "0"
        assert df["number_of_photos_clean"].tolist()[0] == "0"

        # message URLs removed and spaces collapsed
        assert "http" not in df.loc[0, "message_clean"]
        assert df.loc[1, "message_clean"] == "tudo bem"

        # *_length consistent with *_clean
        for col in ["name", "rating", "number_of_photos", "message"]:
            assert df[f"{col}_length"].equals(df[f"{col}_clean"].str.len())

    def test_prepare_columns_missing_column_raises_keyerror(self):
        """If the field does not exist in DataFrame, raise KeyError."""
        df = pd.DataFrame({"wrong": ["x"]})
        with pytest.raises(KeyError):
            pp.prepare_columns(df, "name")

    # -----------------------------
    # run (end-to-end)
    # -----------------------------
    def test_run_missing_input_raises_filenotfound(self, tmp_path: Path):
        """run() must raise FileNotFoundError for a non-existent input file."""
        with pytest.raises(FileNotFoundError):
            pp.run(str(tmp_path / "nope.csv"), str(tmp_path / "out.csv"))

    def test_run_end_to_end_creates_output_and_transforms(self, tmp_path: Path):
        """
        run() should read a pipe-delimited CSV, apply the pipeline, and write a regular CSV.
        Also validates the specific rule for 'number_of_photos' and URL removal in 'message'.
        """
        in_csv = tmp_path / "in.csv"
        out_csv = tmp_path / "out" / "out.csv"  # also tests mkdir(parents=True)

        rows = [
            {"name": "  Peter  ", "rating": " 5 ", "number_of_photos": "", "message": "Hi http://spam"},
            {"name": "Martha", "rating": "4", "number_of_photos": "2", "message": "   I am fine "},
            # A bad row with too many fields will be skipped by on_bad_lines="skip"
            # (we don't need to add it explicitly since DictWriter enforces header, but the reader is robust anyway)
        ]
        self._write_pipe_csv(in_csv, rows)

        pp.run(str(in_csv), str(out_csv))

        assert out_csv.exists(), "Output CSV was not created."

        out_df = pd.read_csv(out_csv)
        expected_cols = {
            "name", "rating", "number_of_photos", "message",
            "name_clean", "name_length",
            "rating_clean", "rating_length",
            "number_of_photos_clean", "number_of_photos_length",
            "message_clean", "message_length",
        }
        assert expected_cols.issubset(set(out_df.columns))

        # key transformations
        assert out_df.loc[0, "name_clean"] == "peter"
        assert out_df.loc[1, "name_clean"] == "martha"
        assert out_df.loc[0, "number_of_photos_clean"] in (0, 0.0)
        assert "http" not in out_df.loc[0, "message_clean"]

    # -----------------------------
    # CLI (__main__)
    # -----------------------------
    def test_cli_subprocess_executes_script(self, tmp_path: Path):
        """
        Execute the module as a script via subprocess:
          python preprocess.py --input <in> --output <out>
        This covers the __main__ block and argparse wiring.
        """
        script_path = Path(pp.__file__).resolve()

        in_csv = tmp_path / "in.csv"
        out_csv = tmp_path / "out.csv"
        rows = [
            {"name": "  Ana  ", "rating": " 5 ", "number_of_photos": "", "message": "Hi http://spam"},
            {"name": "CARLOS", "rating": "4", "number_of_photos": "2", "message": "   Tudo   bem "},
        ]
        self._write_pipe_csv(in_csv, rows)

        proc = subprocess.run(
            [sys.executable, str(script_path), "--input", str(in_csv), "--output", str(out_csv)],
            capture_output=True,
            text=True,
            check=True,
        )

        assert out_csv.exists()
        out_df = pd.read_csv(out_csv)
        assert out_df.loc[0, "name_clean"] == "ana"
        assert "http" not in out_df.loc[0, "message_clean"]
        assert "[preprocess] saved ->" in proc.stdout
