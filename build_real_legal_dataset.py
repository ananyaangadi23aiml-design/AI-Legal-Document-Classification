"""
build_real_legal_dataset.py

Creates:
dataset/
    affidavit/          20 real/public PDF documents
    agreement/          20 real/public PDF documents
    contract/           20 real/public PDF documents
    court_judgement/    20 real/public PDF documents
    legal_notice/       20 real/public PDF documents

The script does NOT generate synthetic legal documents.
It searches public web sources, downloads PDFs, checks that they are real PDFs,
extracts text, scores them for the requested category, removes duplicates,
and stops with a clear report if it cannot reach 20 in a category.

Run from:
C:\\Users\\DELL\\Desktop\\LegalDocumentProject

Requirements:
    py -3.11 -m pip install requests beautifulsoup4 pymupdf

Then:
    py -3.11 build_real_legal_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF is missing.")
    print("Run: py -3.11 -m pip install pymupdf")
    raise SystemExit(1)


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
TARGET = 20
TIMEOUT = 25

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/140.0 Safari/537.36"
    )
}

CATEGORIES = {
    "affidavit": {
        "queries": [
            '"affidavit" filetype:pdf site:gov.in',
            '"affidavit" filetype:pdf site:nic.in',
            '"affidavit" filetype:pdf India government',
            '"affidavit form" filetype:pdf India',
            '"affidavit proforma" filetype:pdf India',
        ],
        "strong": [
            "AFFIDAVIT", "SOLEMNLY AFFIRM", "DEPONENT",
            "SWORN", "VERIFICATION", "NOTARY",
        ],
        "weak": ["deponent", "oath", "affirmed"],
    },
    "agreement": {
        "queries": [
            '"agreement" filetype:pdf India contract',
            '"employment agreement" filetype:pdf India',
            '"vendor agreement" filetype:pdf India',
            '"service agreement" filetype:pdf India',
            '"lease agreement" filetype:pdf India',
            '"non disclosure agreement" filetype:pdf India',
        ],
        "strong": [
            "AGREEMENT", "PARTIES", "WHEREAS", "IN WITNESS WHEREOF",
            "TERMS AND CONDITIONS", "EFFECTIVE DATE",
        ],
        "weak": ["party", "signature", "confidentiality", "termination"],
    },
    "contract": {
        "queries": [
            '"contract" filetype:pdf India legal',
            '"contract of indemnity" filetype:pdf India',
            '"contract agreement" filetype:pdf India',
            '"sale of goods contract" filetype:pdf India',
            '"employment contract" filetype:pdf India',
            '"Indian Contract Act" filetype:pdf',
        ],
        "strong": [
            "CONTRACT", "CONSIDERATION", "INDEMNITY",
            "GUARANTEE", "BREACH", "CONTRACT ACT",
        ],
        "weak": ["obligation", "liability", "termination", "party"],
    },
    "court_judgement": {
        "queries": [
            '"judgment" filetype:pdf site:judgments.ecourts.gov.in',
            '"judgement" filetype:pdf site:ecourts.gov.in',
            '"judgment" filetype:pdf "High Court of" India',
            '"judgement" filetype:pdf "High Court" India',
            '"order" filetype:pdf "High Court" India legal',
        ],
        "strong": [
            "JUDGMENT", "JUDGEMENT", "ORDER", "PETITIONER",
            "RESPONDENT", "HIGH COURT", "SUPREME COURT",
            "CORAM", "PRAYER",
        ],
        "weak": ["appeal", "petition", "learned counsel", "disposed"],
    },
    "legal_notice": {
        "queries": [
            '"legal notice" filetype:pdf India',
            '"legal notice" filetype:pdf site:livelaw.in',
            '"legal notice" filetype:pdf site:hindi.livelaw.in',
            '"legal notice" filetype:pdf advocate India',
            '"statutory legal notice" filetype:pdf India',
            '"notice under section 138" filetype:pdf India',
        ],
        "strong": [
            "LEGAL NOTICE", "UNDER INSTRUCTIONS", "MY CLIENT",
            "HEREBY SERVE", "TAKE NOTICE", "ADVOCATE",
            "WITHOUT PREJUDICE",
        ],
        "weak": ["notice", "demand", "registered post", "speed post"],
    },
}

# A few known genuine/public sources. These are only seeds; the script also searches.
SEED_URLS = {
    "affidavit": [
        "https://www.mea.gov.in/Images/pdf1/NORI_Affidavit_proforma_new.pdf",
    ],
    "agreement": [],
    "contract": [],
    "court_judgement": [],
    "legal_notice": [
        "https://www.livelaw.in/pdf_upload/final-legal-notice--410228.pdf",
        "https://hindi.livelaw.in/pdf_upload/pdf_upload-369738.pdf",
    ],
}


def clean_name(name: str) -> str:
    name = unquote(name)
    name = re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name).strip("._")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:150]


def search_duckduckgo(query: str, max_results: int = 30) -> list[str]:
    url = "https://html.duckduckgo.com/html/?q=" + quote(query)
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        print(f"    search failed: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.select("a.result__a"):
        href = a.get("href", "")
        if not href:
            continue
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("/"):
            href = urljoin("https://html.duckduckgo.com", href)
        if href.startswith("http"):
            links.append(href)
        if len(links) >= max_results:
            break
    return links


def looks_like_pdf_response(response: requests.Response) -> bool:
    ctype = (response.headers.get("content-type") or "").lower()
    prefix = response.content[:5]
    return prefix == b"%PDF-" or "application/pdf" in ctype


def download_pdf(url: str) -> bytes | None:
    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT,
            allow_redirects=True,
        )
        if r.status_code != 200:
            return None
        if not looks_like_pdf_response(r):
            return None
        if len(r.content) < 10_000:
            return None
        return r.content
    except Exception:
        return None


def extract_text(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = min(len(doc), 5)
        text = "\n".join(doc[i].get_text() for i in range(pages))
        doc.close()
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def score(text: str, category: str) -> int:
    t = text.upper()
    rules = CATEGORIES[category]
    s = 0
    for phrase in rules["strong"]:
        if phrase.upper() in t:
            s += 4
    for phrase in rules["weak"]:
        if phrase.upper() in t:
            s += 1
    return s


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def candidate_urls(category: str) -> list[str]:
    found = list(SEED_URLS.get(category, []))
    seen = set(found)

    for q in CATEGORIES[category]["queries"]:
        print(f"  Searching: {q}")
        for url in search_duckduckgo(q, max_results=20):
            # Prefer direct PDFs, but keep pages that may contain PDF links.
            if url not in seen:
                seen.add(url)
                found.append(url)
        time.sleep(0.8)

    return found


def expand_page_for_pdfs(url: str) -> list[str]:
    if ".pdf" in url.lower():
        return [url]
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for a in soup.find_all("a", href=True):
            href = urljoin(r.url, a["href"])
            if ".pdf" in href.lower():
                out.append(href)
        return out[:20]
    except Exception:
        return []


def collect_category(category: str, used_hashes: set[str]) -> list[dict]:
    folder = DATASET / category
    folder.mkdir(parents=True, exist_ok=True)

    # Remove only files created by this script in this target folder.
    for p in folder.glob("*.pdf"):
        p.unlink()

    urls = candidate_urls(category)
    candidates = []
    seen_urls = set()

    for url in urls:
        for pdf_url in expand_page_for_pdfs(url):
            if pdf_url not in seen_urls:
                seen_urls.add(pdf_url)
                candidates.append(pdf_url)

    scored = []

    for i, url in enumerate(candidates, 1):
        if len(scored) >= TARGET:
            break

        data = download_pdf(url)
        if not data:
            continue

        digest = sha256(data)
        if digest in used_hashes:
            continue

        text = extract_text(data)
        if len(text) < 80:
            continue

        s = score(text, category)
        if s < 4:
            continue

        parsed = urlparse(url)
        host = parsed.netloc.lower()

        # Save candidate for sorting by confidence.
        scored.append({
            "url": url,
            "host": host,
            "score": s,
            "hash": digest,
            "data": data,
            "text_len": len(text),
        })

        print(
            f"    accepted candidate {len(scored):02d}: "
            f"score={s}, {host}, {url[:100]}"
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    rows = []
    for idx, item in enumerate(scored[:TARGET], 1):
        used_hashes.add(item["hash"])

        parsed = urlparse(item["url"])
        base = Path(parsed.path).name or f"{category}_{idx}.pdf"
        filename = f"{idx:02d}_{clean_name(base)}"
        path = folder / filename
        path.write_bytes(item["data"])

        rows.append({
            "category": category,
            "filename": filename,
            "source_url": item["url"],
            "source_host": item["host"],
            "keyword_score": item["score"],
            "text_characters_first_5_pages": item["text_len"],
            "sha256": item["hash"],
        })

    return rows


def main():
    print("=" * 72)
    print("REAL LEGAL PDF DATASET BUILDER")
    print("Target: 20 PDFs × 5 categories = 100 PDFs")
    print("=" * 72)
    print()
    print("Project folder:", ROOT)
    print("Dataset folder:", DATASET)
    print()

    # Protect the user's existing dataset automatically.
    backup = ROOT / "dataset_old_before_real_collection"
    if DATASET.exists() and any(DATASET.iterdir()):
        if backup.exists():
            # Pick a non-conflicting backup name.
            n = 2
            while (ROOT / f"dataset_old_before_real_collection_{n}").exists():
                n += 1
            backup = ROOT / f"dataset_old_before_real_collection_{n}"
        print("Existing dataset found.")
        print(f"Backing it up to: {backup}")
        DATASET.rename(backup)

    DATASET.mkdir(exist_ok=True)
    used_hashes: set[str] = set()
    all_rows = []
    failures = []

    for category in CATEGORIES:
        print()
        print("-" * 72)
        print(f"COLLECTING: {category.upper()}")
        print("-" * 72)

        rows = collect_category(category, used_hashes)
        all_rows.extend(rows)

        if len(rows) < TARGET:
            failures.append((category, len(rows)))
            print(
                f"WARNING: only {len(rows)}/{TARGET} verified candidates "
                f"were found for {category}."
            )
        else:
            print(f"OK: {len(rows)}/{TARGET}")

    csv_path = ROOT / "dataset_sources.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "filename",
                "source_url",
                "source_host",
                "keyword_score",
                "text_characters_first_5_pages",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=" * 72)
    print("FINAL REPORT")
    print("=" * 72)

    total = 0
    for category in CATEGORIES:
        count = len(list((DATASET / category).glob("*.pdf")))
        total += count
        print(f"{category:18s}: {count:2d}/20")

    print(f"{'TOTAL':18s}: {total:3d}/100")
    print()
    print("Source log:", csv_path)

    if failures:
        print()
        print("The script did NOT pretend incomplete categories were complete.")
        print("Categories needing more verified PDFs:")
        for category, count in failures:
            print(f"  - {category}: {count}/20")
        print()
        print(
            "If this happens, send the contents of this report to ChatGPT. "
            "Do not use the incomplete dataset for BERT training yet."
        )
        raise SystemExit(3)

    print()
    print("SUCCESS: 100 verified public PDFs collected.")
    zip_path = ROOT / "legal_dataset_100.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdf in sorted(DATASET.rglob("*.pdf")):
            zf.write(pdf, pdf.relative_to(ROOT))
        zf.write(csv_path, csv_path.relative_to(ROOT))
    print("ZIP created:", zip_path)
    print("Source manifest included:", csv_path)
    print("You can now use the dataset folder for BERT training.")


if __name__ == "__main__":
    main()
