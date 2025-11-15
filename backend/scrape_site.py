import os
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
import urllib.robotparser as robotparser
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup
import html2text
from tqdm import tqdm

BASE_DOMAIN = "www.uwo.ca"
START_URL = "https://www.uwo.ca/index.html"
USER_AGENT = "uwo-ai-chatbot/1.0 (course-project; contact: your_email@uwo.ca)"
MAX_PAGES = 500  # increase if you want more pages

HTML_DIR = "scraped/html"
TEXT_DIR = "scraped/text"

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


def init_robot_parser():
    rp = robotparser.RobotFileParser()
    rp.set_url("https://uwo.ca/robots.txt")
    try:
        rp.read()
    except Exception as e:
        print("Warning: Could not read robots.txt:", e)
    return rp


def clean_html_to_text(html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    return h.handle(html)


def is_same_domain(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == BASE_DOMAIN

def normalize_url(current_url: str, link: str) -> Optional[str]:
    if link.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None

    abs_url = urljoin(current_url, link)
    abs_url, _ = urldefrag(abs_url)

    parsed = urlparse(abs_url)

    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.netloc != BASE_DOMAIN:
        return None

    if any(parsed.path.lower().endswith(ext) for ext in [
        ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".doc", ".docx"
    ]):
        return None

    return abs_url


def filename_from_url(url: str, ext: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "home"
    fname = (parsed.netloc + path).replace("/", "_")
    return fname + ext


def scrape_page(url: str, session: requests.Session) -> Optional[Tuple[str, str]]:
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = session.get(url, headers=headers, timeout=10)
    except Exception as e:
        print(f"[ERROR] Request failed for {url}: {e}")
        return None

    if resp.status_code != 200:
        print(f"[WARN] Non-200 ({resp.status_code}) for {url}")
        return None

    if "text/html" not in resp.headers.get("Content-Type", ""):
        return None

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "nav", "footer"]):
        tag.extract()

    cleaned_html = str(soup)
    text = clean_html_to_text(cleaned_html)

    return cleaned_html, text


def crawl_site():
    rp = init_robot_parser()

    visited = set()
    queue = deque([START_URL])
    session = requests.Session()

    pbar = tqdm(total=MAX_PAGES, desc="Crawling uwo.ca")

    while queue and len(visited) < MAX_PAGES:
        url = queue.popleft()
        if url in visited:
            continue

        visited.add(url)

        try:
            if not rp.can_fetch(USER_AGENT, url):
                print(f"[ROBOTS] Disallowed: {url}")
                continue
        except:
            continue

        result = scrape_page(url, session)
        if result is None:
            continue

        html, text = result

        html_fname = filename_from_url(url, ".html")
        text_fname = filename_from_url(url, ".txt")

        with open(os.path.join(HTML_DIR, html_fname), "w", encoding="utf-8") as f:
            f.write(html)

        with open(os.path.join(TEXT_DIR, text_fname), "w", encoding="utf-8") as f:
            f.write(text)

        pbar.update(1)

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            new_url = normalize_url(url, a["href"])
            if new_url and new_url not in visited:
                queue.append(new_url)

        time.sleep(0.5)

    pbar.close()
    print(f"Done. Scraped {len(visited)} pages.")


if __name__ == "__main__":
    crawl_site()
