from pathlib import Path
import csv
import re
import pymupdf
import pytesseract
from PIL import Image

ROOT = Path.cwd()
SOURCE = ROOT / "public_legal_source_download"
OUT = ROOT / "ocr_candidate_review.csv"

# Focus only on scanned PDFs that may help with the target classes.
TARGETS = [
    "NDA template.pdf",
    "Niranjan Shankar Golikari case.pdf",
    "wrongful termination employment.pdf",
    "wrongful termination employment case 2.pdf",
    "arbitration clause.pdf",
    "arbitration clause 2.pdf",
]

if not SOURCE.exists():
    raise SystemExit(
        f"Could not find {SOURCE}\n"
        "Run this script from your LegalDocumentProject folder."
    )

rows = []
for filename in TARGETS:
    path = SOURCE / filename
    if not path.exists():
        rows.append({
            "filename": filename,
            "status": "FILE NOT FOUND",
            "pages": 0,
            "ocr_characters": 0,
            "text_preview": "",
            "error": f"Expected file at {path}",
        })
        continue

    all_text = []
    errors = []
    page_count = 0
    try:
        with pymupdf.open(path) as doc:
            page_count = len(doc)
            for page_no, page in enumerate(doc):
                # If the page already has meaningful embedded text, use it.
                embedded = page.get_text("text").strip()
                if len(embedded) >= 40:
                    text = embedded
                else:
                    try:
                        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text = pytesseract.image_to_string(image, config="--psm 3")
                    except Exception as e:
                        text = ""
                        errors.append(f"page {page_no + 1}: {type(e).__name__}: {e}")
                all_text.append(text)
    except Exception as e:
        rows.append({
            "filename": filename,
            "status": "PDF OPEN ERROR",
            "pages": page_count,
            "ocr_characters": 0,
            "text_preview": "",
            "error": f"{type(e).__name__}: {e}",
        })
        continue

    normalized = re.sub(r"\s+", " ", "\n".join(all_text)).strip()
    status = "OCR/text extracted" if normalized else "No text extracted"
    rows.append({
        "filename": filename,
        "status": status,
        "pages": page_count,
        "ocr_characters": len(normalized),
        "text_preview": normalized[:2500],
        "error": " | ".join(errors),
    })

with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["filename", "status", "pages", "ocr_characters", "text_preview", "error"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Finished checking {len(TARGETS)} PDFs.")
print(f"Report saved to: {OUT}")
print("No PDFs were moved, renamed, or deleted.")
