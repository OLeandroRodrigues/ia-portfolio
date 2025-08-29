# src/google_reviews/cli.py
from __future__ import annotations

import argparse
from pathlib import Path

from .scraping.scraper_reviews import GoogleReviewsScraper, ScraperConfig
from .parsing.parser_reviews import ReviewsParser, ParserConfig


def run_pipeline(url: str,
                 max_comments: int,
                 headless: bool,
                 dirty_file: Path,
                 clean_file: Path) -> None:
    """Run scraper + parser pipeline and save both raw and clean outputs."""
    # --- Scraping phase ---
    s_cfg = ScraperConfig(
        max_comments=max_comments,
        headless=headless,
        # Ensure the scraper writes exactly to dirty_file
        output_file_name=dirty_file.name
    )
    with GoogleReviewsScraper(url, s_cfg) as scraper:
        raw_path = scraper.run()  # scraper resolves path inside data/ by default

        # If user passed a custom path outside default, move/rename after scrape
        # (optional behavior; keep simple: if names differ, ensure final path == dirty_file)
        if raw_path != dirty_file:
            dirty_file.parent.mkdir(parents=True, exist_ok=True)
            dirty_file.write_text(raw_path.read_text(encoding=s_cfg.encoding), encoding=s_cfg.encoding)
            raw_path = dirty_file

    # --- Parsing phase ---
    parser = ReviewsParser(ParserConfig())
    df = parser.parse_file(raw_path)
    clean_file.parent.mkdir(parents=True, exist_ok=True)
    parser.save_csv(df, clean_file)

    print(f"[OK] Raw:   {raw_path}")
    print(f"[OK] Clean: {clean_file}")


def main(argv: list[str] | None = None) -> int:
    """Package CLI entrypoint."""
    p = argparse.ArgumentParser(description="Google Reviews: scrape + parse pipeline")
    p.add_argument("--url", required=True, help="Target Google Maps/Reviews URL.")
    p.add_argument("--max-comments", type=int, default=1600, help="Max number of comments to fetch.")
    p.add_argument("--headless", action="store_true", help="Run Chrome in headless mode.")
    p.add_argument("--dirty-file", type=Path, default=Path("data/raw/dirty-data-google-reviews.txt"),
                   help="Path for raw scraped text file.")
    p.add_argument("--clean-file", type=Path, default=Path("data/processed/clean-data-google-reviews.csv"),
                   help="Path for processed CSV file.")
    args = p.parse_args(argv)

    try:
        run_pipeline(
            url=args.url,
            max_comments=args.max_comments,
            headless=args.headless,
            dirty_file=args.dirty_file,
            clean_file=args.clean_file,
        )
        return 0
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())