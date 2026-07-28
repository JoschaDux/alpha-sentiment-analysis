from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_filing_text(path: str | Path) -> str:
    raw = Path(path).read_text(errors="ignore")
    if "<html" in raw[:1000].lower() or "<document" in raw[:5000].lower():
        return html_to_text(raw)
    return re.sub(r"\s+", " ", raw).strip()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
