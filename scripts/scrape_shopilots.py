import argparse
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    import requests


def fetch_with_playwright(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        # Wait a bit longer for dynamic content
        time.sleep(3)
        # If there's a hash fragment, try to wait for it to load
        parsed = urlparse(url)
        if parsed.fragment:
            # Wait for the fragment to be visible or wait for potential lazy loading
            try:
                page.wait_for_timeout(2000)
            except:
                pass
        content = page.content()
        browser.close()
        return content


def fetch_with_requests(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ShopilotsCrawler/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    
    text_parts = []
    body = soup.find("body")
    if body:
        for elem in body.stripped_strings:
            line = " ".join(str(elem).split())
            if len(line) >= 2 and not line.startswith("×"):
                text_parts.append(line)
    
    return "\n".join(text_parts)


def save_markdown(out_dir: Path, url: str, content: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    path_part = parsed.path.strip("/").replace("/", "-") or "home"
    fragment = parsed.fragment.replace("#", "") if parsed.fragment else ""
    if fragment:
        slug = f"{path_part}-{fragment}" if path_part != "home" else fragment
    else:
        slug = path_part
    fname = f"{slug}.md"
    path = out_dir / fname
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    return path


def run(base_url: str, paths: list[str], out_dir: Path, delay: float = 1.0) -> None:
    for p in paths:
        url = urljoin(base_url, p)
        try:
            if PLAYWRIGHT_AVAILABLE:
                print(f"Fetching (with browser): {url}")
                html = fetch_with_playwright(url)
            else:
                print(f"Fetching (basic): {url}")
                html = fetch_with_requests(url)
            
            text = extract_text(html)
            if not text or len(text.strip()) < 10:
                print(f"Warning: Little content extracted from {url}")
            
            md = f"Source: {url}\n\n{text}"
            out_path = save_markdown(out_dir, url, md)
            print(f"Saved: {out_path} ({len(text)} chars)")
            time.sleep(delay)
        except Exception as e:
            print(f"Error fetching {url}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape selected Shopilots pages to markdown.")
    parser.add_argument("--base", default="https://shopilots.com", help="Base site URL")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=["/", "/#products", "/#integrations", "/#contact", "/blog"],
        help="List of paths to fetch",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "docs" / "shopilots_site"),
        help="Output directory for markdown files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out)
    run(args.base, args.paths, out_dir)


