from pathlib import Path
import csv
import re
import pymupdf

ROOT = Path.cwd()
SOURCE = ROOT / "public_legal_source_download"
OUT = ROOT / "public_pdf_screening_report.csv"

if not SOURCE.exists():
    raise SystemExit(
        f"Could not find: {SOURCE}\n"
        "Run this script from your LegalDocumentProject folder."
    )

# These are screening clues, not automatic final labels.
RULES = {
    "affidavit": [
        r"\baffidavit\b", r"\bsolemnly affirm", r"\bsworn before\b",
        r"\bdeponent\b"
    ],
    "agreement": [
        r"\bagreement\b", r"\bnon[- ]disclosure\b", r"\bvendor agreement\b",
        r"\bemployment agreement\b", r"\bparties agree\b"
    ],
    "contract": [
        r"\bcontract\b", r"\bconsideration\b", r"\bcontracting parties\b",
        r"\bterms and conditions\b"
    ],
    "court_judgement": [
        r"\bjudgment\b", r"\bjudgement\b", r"\bsupreme court of india\b",
        r"\bhigh court\b", r"\bversus\b", r"\bpetition(er)?\b"
    ],
    "legal_notice": [
        r"\blegal notice\b", r"\bnotice is hereby given\b",
        r"\byou are hereby called upon\b", r"\bcease and desist\b"
    ],
}

EXCLUDE_HINTS = [
    (r"\bact,?\s*\d{4}\b|\bact of \d{4}\b", "Likely statute / Act — exclude from five classes"),
    (r"\bclause type\b|\brisky patterns\b", "Likely clause guide / knowledge-base document"),
    (r"\bpractice note\b|\btermination india\b", "May be a practice note / article; inspect before labeling"),
]

def matches(patterns, text):
    return sum(bool(re.search(p, text, flags=re.I)) for p in patterns)

rows = []
for path in sorted(SOURCE.rglob("*.pdf")):
    extracted = ""
    page_count = 0
    error = ""
    try:
        with pymupdf.open(path) as doc:
            page_count = len(doc)
            chunks = []
            for page in doc:
                chunks.append(page.get_text("text"))
                if sum(len(c) for c in chunks) >= 18000:
                    break
            extracted = "\n".join(chunks)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    normalized = re.sub(r"\s+", " ", extracted).strip()
    if normalized:
        candidate_scores = {
            category: matches(patterns, normalized)
            for category, patterns in RULES.items()
        }
        best_score = max(candidate_scores.values(), default=0)
        best = [k for k, v in candidate_scores.items() if v == best_score and v > 0]
        suggested = "; ".join(best) if best else "manual review"
    else:
        candidate_scores = {category: 0 for category in RULES}
        suggested = "OCR/manual review needed" if not error else "could not read PDF"

    exclusion = ""
    for pattern, reason in EXCLUDE_HINTS:
        if re.search(pattern, path.name + " " + normalized[:4000], flags=re.I):
            exclusion = reason
            break

    rows.append({
        "filename": path.name,
        "relative_path": str(path.relative_to(ROOT)),
        "pages": page_count,
        "characters_extracted": len(normalized),
        "suggested_category_clues_only": suggested,
        "affidavit_hits": candidate_scores["affidavit"],
        "agreement_hits": candidate_scores["agreement"],
        "contract_hits": candidate_scores["contract"],
        "court_judgement_hits": candidate_scores["court_judgement"],
        "legal_notice_hits": candidate_scores["legal_notice"],
        "exclusion_warning": exclusion,
        "extracted_text_preview": normalized[:900],
        "error": error,
    })

fields = list(rows[0].keys()) if rows else ["filename", "relative_path"]
with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Screened {len(rows)} PDFs.")
print(f"Report saved to: {OUT}")
print("This is a screening aid, not verified ground-truth labels.")
print("No source PDFs were moved, renamed, or deleted.")
