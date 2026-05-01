"""Headless-Chrome screenshot + DevTools spot-check probe for the Streamlit UI.

Drives the running app at ``http://localhost:8505`` and writes:
  - docs/screenshots/01_empty_explore.png
  - docs/screenshots/02_empty_predict.png
  - docs/screenshots/03_provider_openai.png
  - docs/screenshots/devtools_values.json   (computed CSS values)

Usage:
    python scripts/ui_screenshot.py
    (assumes streamlit is already running on :8505)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


URL = "http://localhost:8505"
OUT = Path("docs/screenshots")
OUT.mkdir(parents=True, exist_ok=True)


def _driver() -> webdriver.Chrome:
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    opts = Options()
    opts.binary_location = chrome_path
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1600,1000")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-sandbox")
    return webdriver.Chrome(options=opts, service=Service())


def _wait_for_app(driver, timeout: int = 25) -> None:
    """Wait until our brand element is visible — confirms our CSS landed."""
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".hl-brand"))
    )
    # Give the @import-loaded font a beat to land.
    time.sleep(2)


def _devtools_values(driver) -> dict:
    """Pull the computed CSS values the audit asked us to verify."""
    body_bg = driver.execute_script(
        "return getComputedStyle(document.body).backgroundColor"
    )
    body_font = driver.execute_script(
        "return getComputedStyle(document.body).fontFamily"
    )
    # The actual visible UI text lives inside .stApp descendants — measure
    # what users would see, not just the body root.
    brand_font = driver.execute_script(
        "const e = document.querySelector('.hl-brand');"
        "return e ? getComputedStyle(e).fontFamily : null"
    )
    stat_font = driver.execute_script(
        "const e = document.querySelector('.hl-stat code');"
        "return e ? getComputedStyle(e).fontFamily : null"
    )
    brand_color = driver.execute_script(
        "const e = document.querySelector('.hl-brand');"
        "return e ? getComputedStyle(e).color : null"
    )
    iter_card_present = driver.execute_script(
        "return !!document.querySelector('.hl-iter-card')"
    )
    sidebar_bg = driver.execute_script(
        "const e = document.querySelector('[data-testid=\"stSidebar\"]');"
        "return e ? getComputedStyle(e).backgroundColor : null"
    )
    deploy_button = driver.execute_script(
        "const e = document.querySelector('[data-testid=\"stDeployButton\"]');"
        "return e ? getComputedStyle(e).display : 'absent'"
    )
    return {
        "body_background": body_bg,
        "body_fontFamily": body_font,
        "brand_fontFamily": brand_font,
        "stat_fontFamily": stat_font,
        "brand_color": brand_color,
        "sidebar_background": sidebar_bg,
        "iteration_card_present": iter_card_present,
        "deploy_button_display": deploy_button,
    }


def _screenshot(driver, name: str) -> Path:
    out = OUT / name
    driver.save_screenshot(str(out))
    return out


def _toggle_predict_mode(driver) -> None:
    """Click the Predict radio in the sidebar form."""
    driver.execute_script(
        "for (const lbl of document.querySelectorAll('label')) {"
        "  const t = (lbl.textContent || '').trim().toLowerCase();"
        "  if (t === 'predict') { lbl.click(); break; }"
        "}"
    )
    time.sleep(2)  # give Streamlit a rerun


def _toggle_provider_openai(driver) -> None:
    driver.execute_script(
        "for (const lbl of document.querySelectorAll('label')) {"
        "  const t = (lbl.textContent || '').trim().toLowerCase();"
        "  if (t === 'openai') { lbl.click(); break; }"
        "}"
    )
    time.sleep(2)


def main() -> int:
    driver = _driver()
    try:
        # 1. Empty state — Explore mode default.
        driver.get(URL)
        _wait_for_app(driver)
        _screenshot(driver, "01_empty_explore.png")
        values_explore = _devtools_values(driver)

        # 2. Predict mode toggled.
        _toggle_predict_mode(driver)
        _screenshot(driver, "02_empty_predict.png")

        # 3. Provider radio flipped to OpenAI.
        _toggle_provider_openai(driver)
        _screenshot(driver, "03_provider_openai.png")
        values_openai = _devtools_values(driver)

        report = {
            "url": URL,
            "explore_mode": values_explore,
            "openai_provider": values_openai,
        }
        (OUT / "devtools_values.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    finally:
        driver.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
