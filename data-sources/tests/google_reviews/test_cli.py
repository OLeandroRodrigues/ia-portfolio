# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

# Import the module under test.
# Make sure to run tests with: PYTHONPATH=src pytest -q
from google_reviews import cli as cli_mod


# ---------------------------
# Fakes (to avoid real scraping/parsing work)
# ---------------------------

class FakeScraperConfig:
    def __init__(self, max_comments: int, headless: bool, output_file_name: str):
        self.max_comments = max_comments
        self.headless = headless
        self.output_file_name = output_file_name
        self.encoding = "utf-8"  # used by cli.run_pipeline to read/write text


class FakeGoogleReviewsScraper:
    """
    Fake context manager for the scraper.
    - Accepts (url, config) in __init__
    - Implements __enter__/__exit__
    - run() returns a Path pointing to the "raw" text file
    """
    def __init__(self, url: str, config: FakeScraperConfig):
        self.url = url
        self.config = config
        self._raw_path: Path | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_raw_path(self, p: Path):
        self._raw_path = p

    def run(self) -> Path:
        assert self._raw_path is not None, "Call set_raw_path(...) in the test before run()."
        return self._raw_path


class FakeParserConfig:
    pass


class FakeReviewsParser:
    """
    Minimal parser stub returning a single-row dataframe and saving CSV.
    """
    def __init__(self, cfg: FakeParserConfig):
        self.cfg = cfg

    def parse_file(self, raw_path: Path) -> pd.DataFrame:
        # Produce a deterministic, simple dataframe:
        return pd.DataFrame(
            [{"name": "ana", "rating": "4.0/5", "number_of_photos": "2", "message": "great"}]
        )

    def save_csv(self, df: pd.DataFrame, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="|", index=False, encoding="utf-8")


# ---------------------------
# Global patch fixture
# ---------------------------

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """
    Patches dependencies inside google_reviews.cli so tests are hermetic:
    - GoogleReviewsScraper -> FakeGoogleReviewsScraper
    - ScraperConfig -> FakeScraperConfig
    - ReviewsParser -> FakeReviewsParser
    - ParserConfig -> FakeParserConfig

    Exposes a simple holder so tests can set the scraper's raw path.
    """
    holder = {"scraper": None}

    def scraper_ctor(url, cfg):
        inst = FakeGoogleReviewsScraper(url, cfg)
        holder["scraper"] = inst
        return inst

    monkeypatch.setattr(cli_mod, "GoogleReviewsScraper", scraper_ctor)
    monkeypatch.setattr(cli_mod, "ScraperConfig", FakeScraperConfig)
    monkeypatch.setattr(cli_mod, "ReviewsParser", FakeReviewsParser)
    monkeypatch.setattr(cli_mod, "ParserConfig", FakeParserConfig)

    yield holder


# ---------------------------
# Tests for run_pipeline
# ---------------------------

def test_run_pipeline_with_copy(tmp_path, patch_dependencies, capsys):
    """
    When the scraper writes to a different path than the user-provided dirty_file,
    the pipeline should copy the content into dirty_file before parsing.
    """
    # Internal raw file produced by the (fake) scraper:
    internal_raw = tmp_path / "data" / "raw" / "internal_default.txt"
    internal_raw.parent.mkdir(parents=True, exist_ok=True)
    internal_raw.write_text("RAW CONTENT", encoding="utf-8")

    # Desired user-specified output paths:
    dirty_file = tmp_path / "custom" / "dirty-data-google-reviews.txt"
    clean_file = tmp_path / "custom" / "clean-data-google-reviews.csv"

    # Configure fake scraper to point to internal_raw:
    # (The instance is created inside run_pipeline; we set after constructor via holder)
    # We trigger the constructor by calling run_pipeline first, then set the path before run() is used:
    # To keep it simple, set after calling run_pipeline is not possible, so ensure the holder is used correctly:
    # -> We can pre-create the instance by forcing a call that constructs it? Not needed:
    # We will set it lazily via a custom wrapper below.
    # Instead, we patch the constructor a second time here to set_raw_path immediately.
    original_ctor = cli_mod.GoogleReviewsScraper

    def ctor_and_set(url, cfg):
        inst = original_ctor(url, cfg)  # returns FakeGoogleReviewsScraper
        inst.set_raw_path(internal_raw)
        return inst

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli_mod, "GoogleReviewsScraper", ctor_and_set)

    try:
        cli_mod.run_pipeline(
            url="https://maps.google.com/fake",
            max_comments=10,
            headless=True,
            dirty_file=dirty_file,
            clean_file=clean_file,
        )
    finally:
        monkeypatch.undo()

    # Artifacts must exist:
    assert dirty_file.exists(), "dirty_file should be created by the pipeline."
    assert clean_file.exists(), "clean_file (CSV) should be created by the parser."

    # Check informative prints:
    out = capsys.readouterr().out
    assert "[OK] Raw:" in out
    assert "[OK] Clean:" in out


def test_run_pipeline_without_copy(tmp_path, patch_dependencies):
    """
    When scraper's raw path equals the dirty_file path, no copy is needed.
    """
    dirty_file = tmp_path / "data" / "raw" / "dirty-data-google-reviews.txt"
    clean_file = tmp_path / "data" / "processed" / "clean-data-google-reviews.csv"
    dirty_file.parent.mkdir(parents=True, exist_ok=True)
    dirty_file.write_text("RAW CONTENT (same path)", encoding="utf-8")

    # Ensure the fake scraper returns exactly dirty_file
    original_ctor = cli_mod.GoogleReviewsScraper

    def ctor_and_set(url, cfg):
        inst = original_ctor(url, cfg)
        inst.set_raw_path(dirty_file)
        return inst

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli_mod, "GoogleReviewsScraper", ctor_and_set)
    try:
        cli_mod.run_pipeline(
            url="https://maps.google.com/fake",
            max_comments=5,
            headless=False,
            dirty_file=dirty_file,
            clean_file=clean_file,
        )
    finally:
        monkeypatch.undo()

    assert dirty_file.exists()
    assert clean_file.exists()


# ---------------------------
# Tests for main()
# ---------------------------

def test_main_success(tmp_path, monkeypatch):
    """
    Exercise the CLI entrypoint with custom argv and ensure it returns 0.
    """
    dirty_file = tmp_path / "raw.txt"
    clean_file = tmp_path / "clean.csv"
    internal_raw = tmp_path / "internal.txt"
    internal_raw.write_text("RAW CONTENT INTERNAL", encoding="utf-8")

    # Patch dependencies to our fakes (again here, local to this test)
    monkeypatch.setattr(cli_mod, "ScraperConfig", FakeScraperConfig)
    monkeypatch.setattr(cli_mod, "ReviewsParser", FakeReviewsParser)
    monkeypatch.setattr(cli_mod, "ParserConfig", FakeParserConfig)

    original_ctor = cli_mod.GoogleReviewsScraper

    def ctor_and_set(url, cfg):
        inst = FakeGoogleReviewsScraper(url, cfg)
        inst.set_raw_path(internal_raw)
        return inst

    monkeypatch.setattr(cli_mod, "GoogleReviewsScraper", ctor_and_set)

    argv = [
        "--url", "https://maps.google.com/fake",
        "--max-comments", "3",
        "--headless",
        "--dirty-file", str(dirty_file),
        "--clean-file", str(clean_file),
    ]
    rc = cli_mod.main(argv)
    assert rc == 0
    assert dirty_file.exists()
    assert clean_file.exists()


def test_main_handles_exception(monkeypatch):
    """
    Make run_pipeline raise and verify main returns 1 and prints an error line.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("exploded")

    monkeypatch.setattr(cli_mod, "run_pipeline", boom)
    rc = cli_mod.main(["--url", "http://x", "--dirty-file", "x.txt", "--clean-file", "y.csv"])
    assert rc == 1