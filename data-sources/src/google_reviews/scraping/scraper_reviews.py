# src/google_reviews/scraping/scraper_reviews.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@dataclass(frozen=True)
class ScraperConfig:
    """Immutable configuration for GoogleReviewsScraper."""
    # XPaths / selectors
    scrollable_xpath: str = "//div[@class='RVCQse']"
    comment_xpath: str = "//div[@class='bwb7ce']"
    more_btn_xpaths: tuple[str, ...] = (
        "//div[contains(@class,'OA1nbd')]//a[contains(@class,'MtCSLb') and @role='button']",
        "//a[@role='button' and contains(@aria-label,'Ler mais')]",
        "//a[@role='button' and contains(.,'Mais')]",
    )

    # Scrolling behavior
    max_comments: int = 1600
    scroll_pause: float = 1.2
    scroll_pixels: int = 800
    max_idle_rounds: int = 8

    # WebDriver behavior
    wait_timeout: float = 10.0
    headless: bool = False
    start_maximized: bool = True
    disable_automation_blink: bool = True

    # Output
    output_file_name: str = "dirty-data-google-reviews.txt"
    encoding: str = "utf-8"


class GoogleReviewsScraper:
    """OOP Google Reviews scraper with Selenium Manager (no webdriver-manager required)."""

    def __init__(self, url: str, config: Optional[ScraperConfig] = None, driver: Optional[webdriver.Chrome] = None):
        self.url = url
        self.config = config or ScraperConfig()
        self._driver_external = driver is not None
        self.driver: Optional[webdriver.Chrome] = driver
        self._output_path: Optional[Path] = None

    # -------- Context management to ensure cleanup --------
    def __enter__(self) -> "GoogleReviewsScraper":
        if self.driver is None:
            self.driver = self._build_driver()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # -------- Driver lifecycle --------
    def _build_driver(self) -> webdriver.Chrome:
        """Build a Chrome WebDriver using Selenium Manager to fetch the correct driver."""
        options = Options()
        if self.config.start_maximized:
            options.add_argument("--start-maximized")
        if self.config.disable_automation_blink:
            options.add_argument("--disable-blink-features=AutomationControlled")
        # Prefer headless=new; fallback to classic if needed
        if self.config.headless:
            try:
                options.add_argument("--headless=new")
            except Exception:
                options.add_argument("--headless")

        # Selenium Manager (Selenium >= 4.6) resolves the proper ChromeDriver automatically.
        # Service() without path is enough; you can even omit Service() entirely.
        return webdriver.Chrome(service=Service(), options=options)
        # Alternatively: return webdriver.Chrome(options=options)

    

    def close(self) -> None:
        """Quit the driver if it was created by this instance."""
        if self.driver and not self._driver_external:
            try:
                self.driver.quit()
            finally:
                self.driver = None

    # -------- High-level API --------
    def run(self) -> Path:
        """Open the page, scroll/expand, extract comments, save to file, and return its path."""
        if self.driver is None:
            self.driver = self._build_driver()

        self._open_page()
        container = self._wait_for_container()
        self._scroll_and_expand(container)
        comments = self._extract_comments()

        out = self._resolve_output_path()
        self._save_comments(comments, out)
        return out

    # -------- Steps --------
    def _open_page(self) -> None:
        assert self.driver is not None
        print(f"[INFO] Navigating to: {self.url}")
        self.driver.get(self.url)

    def _wait_for_container(self):
        """Wait for the scrollable container to be present and try to focus it."""
        assert self.driver is not None
        wait = WebDriverWait(self.driver, self.config.wait_timeout)
        try:
            container = wait.until(EC.presence_of_element_located((By.XPATH, self.config.scrollable_xpath)))
            try:
                container.click()
            except Exception:
                pass
            return container
        except TimeoutException:
            raise RuntimeError("Scrollable container not found. Check selectors or page state.")

    def _scroll_and_expand(self, container) -> None:
        """Scroll the reviews list and expand visible 'More' buttons until saturation."""
        assert self.driver is not None
        total_seen = 0
        idle_rounds = 0

        for _ in range(max(1, self.config.max_comments)):
            # Perform gesture scroll; fallback to PAGE_DOWN if needed
            try:
                origin = ScrollOrigin.from_element(container)
                ActionChains(self.driver).scroll_from_origin(origin, 0, self.config.scroll_pixels).perform()
            except Exception:
                container.send_keys(Keys.PAGE_DOWN)

            time.sleep(self.config.scroll_pause)

            # Try to expand any visible "More" snippet buttons
            clicked = self._click_more_in_view()

            # Track number of loaded comments to detect progress
            current = self._count_comments()
            if current <= total_seen and clicked == 0:
                idle_rounds += 1
            else:
                idle_rounds = 0
                total_seen = current

            if total_seen >= self.config.max_comments:
                break
            if idle_rounds >= self.config.max_idle_rounds:
                break

    def _click_more_in_view(self) -> int:
        """Click all currently visible 'More' buttons. Returns the number of clicks performed."""
        assert self.driver is not None
        clicked = 0
        for xp in self.config.more_btn_xpaths:
            for btn in self.driver.find_elements(By.XPATH, xp):
                try:
                    if btn.is_displayed():
                        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                        self.driver.execute_script("arguments[0].click();", btn)
                        clicked += 1
                except (StaleElementReferenceException, ElementClickInterceptedException):
                    continue
                except Exception:
                    continue
        return clicked

    def _count_comments(self) -> int:
        """Return how many comment elements are present right now."""
        assert self.driver is not None
        return len(self.driver.find_elements(By.XPATH, self.config.comment_xpath))

    def _extract_comments(self) -> list[str]:
        """Extract single-line trimmed comment texts."""
        assert self.driver is not None
        elements = self.driver.find_elements(By.XPATH, self.config.comment_xpath)
        comments = [el.text.replace("\n", " ").strip() for el in elements if el.text.strip()]
        return comments[: self.config.max_comments]

    # -------- Output handling --------
    def _resolve_output_path(self) -> Path:
        """
        Resolve output path inside the project's data/ folder:
        project_root = two levels above this file; output at data/<output_file_name>.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        self._output_path = Path(os.path.join(data_dir, self.config.output_file_name))
        return self._output_path

    def _save_comments(self, comments: Sequence[str], outfile: Path) -> None:
        """Persist comments joined by a pipe delimiter (|), same as original behavior."""
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with outfile.open("w", encoding=self.config.encoding) as f:
            f.write(" | ".join(comments))
        print(f"[INFO] Saved {len(comments)} comments to: {outfile}")
