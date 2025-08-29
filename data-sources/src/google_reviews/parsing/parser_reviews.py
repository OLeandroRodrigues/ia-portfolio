# src/google_reviews/parsing/parser_reviews.py

from dataclasses import dataclass, field
import re
import pandas as pd
from pathlib import Path
from typing import Optional, List


@dataclass(frozen=True)
class ParserConfig:
    """Configuration for ReviewsParser (regex, noise filters, options)."""
    noise_patterns: tuple[str, ...] = (
        r"Passe o cursor para reagir",
        r"Ofner\s*-\s*Perdizes\s*\(proprietário\).*?(?=(?:\n|$))",
    )
    header_pattern: str = r"""
        (?P<name>[^|\n]+?)
        \s+
        (?:Local\ Guide·\s*)?
        (?P<reviews>\d+)\s+avaliaç(?:ões|ão)
        (?:·(?P<photos>\d+)\s+fotos)?
    """
    csv_sep: str = "|"
    encoding: str = "utf-8"

    _header_re: re.Pattern = field(init=False, repr=False)
    _noise_res: tuple[re.Pattern, ...] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_header_re",
            re.compile(self.header_pattern, re.IGNORECASE | re.VERBOSE),
        )
        object.__setattr__(
            self,
            "_noise_res",
            tuple(re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.noise_patterns),
        )


class ReviewsParser:
    """OOP parser to extract (name, rating, number_of_photos, message)."""

    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig()

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"\s*\|\s*", " | ", s)
        return s

    def _clean_message(self, s: str) -> str:
        for rx in self.config._noise_res:
            s = rx.sub("", s)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{2,}", "\n", s).strip()
        return s

    @staticmethod
    def _sanitize_field(s: str) -> str:
        return s.replace("|", " / ").strip()

    @staticmethod
    def _extract_rating(text: str) -> str:
        vals: List[float] = []
        for m in re.finditer(r"(\d+(?:[.,]\d+)?)[ ]*/[ ]*5\b", text):
            try:
                vals.append(float(m.group(1).replace(",", ".")))
            except ValueError:
                pass
        return f"{sum(vals)/len(vals):.1f}/5" if vals else ""

    def _find_headers(self, text: str):
        return list(self.config._header_re.finditer(text))

    def parse_text(self, text: str) -> pd.DataFrame:
        text = self._normalize_text(text)
        headers = self._find_headers(text)
        rows = []
        for i, h in enumerate(headers):
            name = h.group("name").strip()
            photos = h.group("photos") or ""
            start = h.end()
            end = headers[i+1].start() if i+1 < len(headers) else len(text)
            body = text[start:end].lstrip(" |").strip()
            rating = self._extract_rating(body)
            body = self._clean_message(body)
            rows.append({
                "name": self._sanitize_field(name),
                "rating": self._sanitize_field(rating),
                "number_of_photos": self._sanitize_field(photos),
                "message": self._sanitize_field(body),
            })
        return pd.DataFrame(rows, columns=["name", "rating", "number_of_photos", "message"])

    def parse_file(self, infile: Path) -> pd.DataFrame:
        raw = infile.read_text(encoding=self.config.encoding)
        return self.parse_text(raw)

    def save_csv(self, df: pd.DataFrame, outfile: Path) -> None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outfile, sep=self.config.csv_sep, index=False, encoding=self.config.encoding)
