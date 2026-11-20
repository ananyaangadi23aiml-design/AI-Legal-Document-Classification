
from pathlib import Path
import fitz
import hashlib
import csv

root = Path("public_legal_source_download")
rows = []
seen_hashes = {}

for pdf in sorted(root.rglob("*.pdf")):
    try:
        content = pdf.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        duplicate = seen_hashes.get(digest, "")

        if not duplicate:
            seen_hashes[digest] = str(pdf)

        with fitz.open(stream=content, filetype="pdf") as doc:
            page_count = len(doc)
            text = "\n".join(page.get_text() for page in doc)

        rows.append({
            "filename": pdf.name,
            "pages": page_count,
            "characters": len(text),
            "duplicate_of": duplicate,
            "text_preview": " ".join(text.split())[:500],
            "source_path": str(pdf),
            "status": "OK" if text.strip() else "No extractable text"
        })

    except Exception as e:
        rows.append({
            "filename": pdf.name,
            "pages": "",
            "characters": "",
            "duplicate_of": "",
            "text_preview": "",
            "source_path": str(pdf),
            "status": f"ERROR: {e}"
        })

fields = [
    "filename", "pages", "characters", "duplicate_of",
    "text_preview", "source_path", "status"
]

with open("pdf_inspection_report.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Inspected {len(rows)} PDFs.")
print(f"Readable PDFs: {sum(r['status'] == 'OK' for r in rows)}")
print("Report: pdf_inspection_report.csv")
