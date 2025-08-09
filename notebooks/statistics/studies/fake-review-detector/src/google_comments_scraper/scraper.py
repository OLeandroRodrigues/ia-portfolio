# google_scraper/comment_scraper.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
from selenium.common.exceptions import StaleElementReferenceException, ElementClickInterceptedException

from webdriver_manager.chrome import ChromeDriverManager

import os
import time

class GoogleCommentScraper:
    def __init__(self, url, output_file_name="dirty-data-google-comments.txt", max_comments=100, scroll_delay=1):
        self.url = url
        self.output_file_name = output_file_name
        self.max_comments = max_comments
        self.scroll_delay = scroll_delay
        self.driver = None

    def _init_driver(self):
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def _scroll_comments(self):
        scrollable_xpath = "//div[@class='RVCQse']"
        scrollable_div = self.driver.find_element(By.XPATH, scrollable_xpath)

        try:
            scrollable_div.click()  # focus on container
        except Exception:
            pass
        time.sleep(0.5)

        total_scrolls = max(1, self.max_comments)
        
        for _ in range(total_scrolls):
            try:
                origin = ScrollOrigin.from_element(scrollable_div)
                ActionChains(self.driver).scroll_from_origin(origin, 0, 800).perform()
            except Exception:
                scrollable_div.send_keys(Keys.PAGE_DOWN)

            time.sleep(self.scroll_delay)

            # Expand “Mais” each interation
            self._click_more_in_view()
    

    
    def _click_more_in_view(self):
        """
        Clica em todos os links 'Mais' visíveis nesta tela.
        DOM típico: <a class="MtCSLb" role="button" aria-label="Ler mais ...">Mais</a>
        """
        xpaths = [
            "//div[contains(@class,'OA1nbd')]//a[contains(@class,'MtCSLb') and @role='button']",
            "//a[@role='button' and contains(@aria-label,'Ler mais')]",
            "//a[@role='button' and contains(.,'Mais')]",  # fallback 
        ]
        clicked = 0
        for xp in xpaths:
            btns = self.driver.find_elements(By.XPATH, xp)
            for b in btns:
                try:
                    if b.is_displayed():
                        # Centers and clicks via JS (evita intercept/overlay)
                        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", b)
                        self.driver.execute_script("arguments[0].click();", b)
                        clicked += 1
                except (StaleElementReferenceException, ElementClickInterceptedException):
                    continue
                except Exception:
                    continue
        return clicked

    def _extract_comments(self):
        comment_xpath = "//div[@class='bwb7ce']"
        elements = self.driver.find_elements(By.XPATH, comment_xpath)

        comments = [el.text.replace("\n", " ").strip() for el in elements if el.text.strip()]
        return comments[:self.max_comments]

    def _save_comments(self, comments):

        # Gets the folder where this .py file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Goes up two levels
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # Path to the "data" folder (already existing)
        data_dir = os.path.join(project_root, "data")

        # Final file path inside the "data" folder
        self.output_file = os.path.join(data_dir, self.output_file_name)

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