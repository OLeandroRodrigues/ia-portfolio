
# 🕵️‍♂️ Fake Review Detector (Study Project)

This repository is a **study project** focused on scraping Google Reviews, parsing the raw text, and preparing data for downstream analysis or ML experiments.

---

## 📂 Project Structure

```
fake-review-detector/
├── fake-review-detector/
│   ├── data/
│   │   ├── dirty-data-google-reviews.txt    # Raw reviews scraped from Google
│   │   └── clean-data-google-reviews.txt    # Parsed, pipe-separated data
│   ├── models/
│   │   └── comment.py                        # (Placeholder for future modeling)
│   ├── notebooks/
│   │   └── usage_example.ipynb               # Example notebook
│   └── src/
│       ├── main.py                           # Entry point to run the scraper (edit the URL inside)
│       ├── google_reviews_scraper/
│       │   └── scraper.py                    # Selenium-based scraper
│       ├── google_reviews_parser/
│       │   └── parse_google_reviews.py       # Parser to extract fields (name, rating, photos, message)
│       ├── requirements.txt                  # Project dependencies
│       ├── README.md
│       └── venv/                             # ❌ Local virtualenv (should not be in git)
├── .gitignore
├── TODO.md
└── README.md                                  # (this file)
```

---

## 🛠️ Requirements

- **Python 3.10+** (tested with 3.12)
- **Google Chrome** installed
- The scraper uses:
  - `selenium`
  - `webdriver-manager` (auto-installs the correct ChromeDriver)

Install Python dependencies from `src/requirements.txt`:

```bash
pip install -r fake-review-detector/src/requirements.txt
```

> If you prefer a virtual environment:
> ```bash
> python -m venv .venv
> source .venv/bin/activate   # Linux/macOS
> .venv\Scripts\activate    # Windows
> pip install -r fake-review-detector/src/requirements.txt
> ```

---

## 🚀 How to Run

### 1) Scrape Google Reviews
Edit the URL inside `src/main.py` to your target Google Business reviews page:

```python
# src/main.py
from google_reviews_scraper.scraper import GoogleReviewsScraper

if __name__ == "__main__":
    url = "https://www.google.com/search?q=<your place>&...#lrd=<ids>"
    scraper = GoogleReviewsScraper(url)
    scraper.run()
```

Run the scraper:

```bash
python fake-review-detector/src/main.py
```

This will create/update `fake-review-detector/data/dirty-data-google-reviews.txt` with the raw text.

> You can control max comments and delay via the class constructor:
> ```python
> GoogleReviewsScraper(url, output_file_name="dirty-data-google-reviews.txt", max_comments=1600, scroll_delay=2)
> ```

### 2) Parse the Raw Text
The parser reads the raw text file and outputs a pipe-delimited (`|`) table with columns:
`name | rating | number_of_photos | message`.

Run:

```bash
python fake-review-detector/src/google_reviews_parser/parse_google_reviews.py
```

This generates `fake-review-detector/data/clean-data-google-reviews.txt`.

---

## 🧪 Notebooks

Use `notebooks/usage_example.ipynb` to explore the cleaned data or test feature engineering ideas.

---

## 🧰 Tips & Conventions

- **Do not commit secrets**. If you ever add environment variables, keep them in a `.env` file and add it to `.gitignore`.
- Consider creating a `requirements.txt` at the **repo root** (mirroring `src/requirements.txt`) if you prefer a single installation command for users.
- For **unit tests**, you could add a `tests/` folder and set up `pytest`.

---


## 📄 License

This is a study project. If you want to make it public-friendly, consider adding an explicit license (e.g., MIT).

---

## ✅ Roadmap / TODO

See **`TODO.md`** for planned improvements and tasks.
