# google_scraper/comment_scraper.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

class GoogleCommentScraper:
    def __init__(self, url, output_file="google-commentts.txt", max_comments=100, scroll_delay=2):
        self.url = url
        self.output_file = output_file
        self.max_comments = max_comments
        self.scroll_delay = scroll_delay
        self.driver = None

    def _init_driver(self):
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def _scroll_comments(self):
        scrollable_xpath = "//div[@class='m6QErb DxyBCb kA9KIf dS8AEf ecceSd']"
        scrollable_div = self.driver.find_element(By.XPATH, scrollable_xpath)
        num_scrolls = 0

        while num_scrolls < self.max_comments // 10:
            self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
            time.sleep(self.scroll_delay)
            num_scrolls += 1

    def _extract_comments(self):
        comment_xpath = "//div[@class='wiI7pd']"
        elements = self.driver.find_elements(By.XPATH, comment_xpath)
        comments = [el.text.replace("\n", " ").strip() for el in elements if el.text.strip()]
        return comments[:self.max_comments]

    def _save_comments(self, comments):
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(" | ".join(comments))

    def run(self):
        self._init_driver()
        print(f"[INFO] Connection: {self.url}")
        self.driver.get(self.url)
        time.sleep(5)

        print("[INFO] Scrooling comments...")
        self._scroll_comments()

        print("[INFO] Extracting comments...")
        comments = self._extract_comments()
        print(f"[INFO] {len(comments)} comments extracts.")

        self._save_comments(comments)
        print(f"[INFO] Comments saved at: {self.output_file}")

        self.driver.quit()