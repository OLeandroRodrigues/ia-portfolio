## 📝 Google Reviews Scraper & Parser

This project provides a **Python package** to automatically scrape Google Reviews from a given place URL and parse them into a clean CSV dataset.  
It uses **Selenium 4 (with Selenium Manager)** for scraping and **Pandas** for parsing/exporting.

---

## 🚀 Features
- 🔍 Scrapes reviews directly from **Google Maps**.
- 📜 Extracts user name, rating, number of photos, and review message.
- 🧹 Cleans noisy text (removes "read more" etc.).
- 📂 Exports to CSV with `|` as delimiter.
- 🕵️ Runs Chrome in **incognito** mode by default.
- 🖥️ CLI (Command Line Interface) ready-to-use.

---

## 📦 Requirements
- Python **3.10+**
- Google Chrome installed (latest version)
- The following dependencies (automatically installed via `pip`):
  - `selenium>=4.18.1`
  - `pandas>=2.0`

---

## 🔧 Installation

Clone this repository and install in **editable mode**:

```bash
git clone https://github.com/OLeandroRodrigues/ia-portfolio.git
cd data-sources

# (Optional) create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\Activate.ps1  # Windows PowerShell

# install package
pip install -e .

```

▶️ Usage
Run via CLI command

After installation you can run the CLI:
```bash
google-reviews --url "https://www.google.com/search?q=loja+ofner+perdizes&oq=loja+ofner+perdizes&gs_lcrp=EgZjaHJvbWUqCggAEAAY4wIYgAQyCggAEAAY4wIYgAQyEAgBEC4YrwEYxwEYgAQYjgUyBwgCEAAY7wUyCggDEAAYgAQYogQyCggEEAAYgAQYogQyCggFEAAYgAQYogTSAQgzNDc4ajBqNKgCALACAQ&sourceid=chrome&ie=UTF-8&sei=0d-QaPLIKLCC5OUPiN2OkAo#lrd=0x94ce57f55b7f4dad:0xb1e756042b056e2d,1,,,," --max-comments 500 --headless
```

--- 

## 📂 Project structure
```
data-sources/
├─ pyproject.toml        # package configuration
├─ src/
│  └─ google_reviews/
│     ├─ __init__.py
│     ├─ cli.py          # CLI entrypoint
│     ├─ parsing/
│     │  └─ parser_reviews.py
│     └─ scraping/
│        └─ scraper_reviews.py
└─ data/
   ├─ raw/               # dirty raw scraped data
   └─ processed/         # clean parsed datasets
```

---
⚠️ **Warning: DOM or URL changes may break scraping**

If the application fails to run or appears to extract **no reviews**, it is likely due to changes in the **Google Maps DOM structure** or an **unsupported URL format**. Google frequently updates the HTML classes and layout of its review elements, which may invalidate the XPath selectors used in the scraper.

📁 You can find the raw scraped file at:
    data/raw/dirty-data-google-reviews.txt
