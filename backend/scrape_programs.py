import requests
from bs4 import BeautifulSoup
import html2text
import os

URL = "https://welcome.uwo.ca/what-can-i-study/undergraduate-programs/index.html"
OUTPUT_PATH = os.path.join("data", "undergraduate_programs.txt")


def clean_html_to_text(html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    return h.handle(html)


def main():
    print(f"Fetching {URL} ...")
    headers = {"User-Agent": "Mozilla/5.0 (WesternChatbot/1.0)"}
    resp = requests.get(URL, headers=headers, timeout=15)

    if resp.status_code != 200:
        print("Failed with status:", resp.status_code)
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # Optional: try to focus on the main content area
    main_content = soup.find("main") or soup  # fallback to whole page

    # Remove noisy elements
    for tag in main_content(["script", "style", "nav", "footer", "noscript"]):
        tag.extract()

    text = clean_html_to_text(str(main_content))

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved cleaned text to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
