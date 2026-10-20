"""
Fast, token-efficient web metadata and content extractor.
Extracts high-signal features (Title, Meta Description, Headings, App/API signals)
to keep token counts low (< 350 tokens) for ultra-fast local LLM inference.
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def clean_text(text: str) -> str:
    """Normalize whitespace and remove excessive noise."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_fast_summary(url: str, timeout: int = 7) -> Dict[str, Any]:
    """
    Fetch homepage and distill only high-signal business indicators.
    Returns a clean dictionary and a prompt-ready markdown/text block.
    """
    result = {
        "url": url,
        "final_url": url,
        "status_code": None,
        "title": "",
        "meta_desc": "",
        "headings": [],
        "has_mobile_apps": False,
        "has_developer_api": False,
        "content_snippet": "",
        "error": None
    }
    
    if not url:
        result["error"] = "Empty URL"
        return result

    if not url.startswith("http://") and not url.startswith("https://"):
        target_url = "https://" + url
    else:
        target_url = url

    def try_fetch(u: str):
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)
        return session.get(u, timeout=timeout, allow_redirects=True, verify=False)

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    response = None
    try:
        response = try_fetch(target_url)
    except Exception as e1:
        # Try parent domain fallback if subdomain failed
        from urllib.parse import urlparse
        parsed = urlparse(target_url)
        parts = parsed.netloc.split(".")
        if len(parts) > 2:
            root_url = f"{parsed.scheme}://{'.'.join(parts[-2:])}"
            try:
                response = try_fetch(root_url)
            except Exception:
                result["error"] = str(e1)
        else:
            result["error"] = str(e1)

    if response is not None:
        result["status_code"] = response.status_code
        result["final_url"] = response.url
        
        if response.status_code < 400:
            try:
                soup = BeautifulSoup(response.text, "html.parser")

                # 1. Page Title
                if soup.title and soup.title.string:
                    result["title"] = clean_text(soup.title.string)

                # 2. Meta description / OpenGraph
                meta_desc = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
                if not meta_desc:
                    meta_desc = soup.find("meta", attrs={"property": "og:description"})
                if meta_desc and meta_desc.get("content"):
                    result["meta_desc"] = clean_text(meta_desc["content"])

                # 3. Headings (H1 & H2)
                headings = []
                for tag in soup.find_all(["h1", "h2"]):
                    h_text = clean_text(tag.get_text())
                    if h_text and len(h_text) > 4 and h_text not in headings:
                        headings.append(h_text)
                    if len(headings) >= 5:
                        break
                result["headings"] = headings

                # 4. Tech / App indicators
                html_lower = response.text.lower()
                if "apps.apple.com" in html_lower or "play.google.com" in html_lower or "app store" in html_lower:
                    result["has_mobile_apps"] = True
                if "api documentation" in html_lower or "developer portal" in html_lower or "docs.api" in html_lower:
                    result["has_developer_api"] = True

                # 5. Extract clean body text (strip nav, scripts, styles, footers)
                for junk in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
                    junk.decompose()

                body_text = clean_text(soup.get_text(" "))
                result["content_snippet"] = body_text[:800]
            except Exception as parse_err:
                result["error"] = f"HTML parsing error: {parse_err}"
        else:
            result["error"] = f"HTTP {response.status_code}"

    return result

def format_prompt_input(summary: Dict[str, Any], company_name: Optional[str] = None) -> str:
    """Format extracted summary into a compact text block for the LLM."""
    lines = []
    if company_name:
        lines.append(f"Company: {company_name}")
    lines.append(f"Website: {summary.get('final_url') or summary.get('url')}")
    
    if summary.get("title"):
        lines.append(f"Title: {summary['title']}")
    if summary.get("meta_desc"):
        lines.append(f"Description: {summary['meta_desc']}")
    if summary.get("headings"):
        lines.append(f"Key Headings: {' | '.join(summary['headings'])}")
    
    tech_signals = []
    if summary.get("has_mobile_apps"):
        tech_signals.append("Proprietary Mobile App links found")
    if summary.get("has_developer_api"):
        tech_signals.append("Developer API / Documentation found")
    if tech_signals:
        lines.append(f"Signals: {', '.join(tech_signals)}")
        
    if summary.get("content_snippet"):
        lines.append(f"Content: {summary['content_snippet']}")
        
    return "\n".join(lines)
