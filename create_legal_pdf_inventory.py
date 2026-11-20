from pathlib import Path
import csv
import hashlib
import re

ROOT = Path(__file__).resolve().parent
SEARCH_DIRS = [
    ROOT / "dataset",
    ROOT / "dataset_synthetic_backup",
    ROOT / "public_legal_source_download",
]
OUTPUT = ROOT / "legal_pdf_inventory.csv"

def sha256(path):
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return ""

def suggested_category(name):
    s = name.lower().replace("_", " ").replace("-", " ")
    if any(x in s for x in ["judgment", "judgement", "case", "supreme court", "high court"]):
        return "court_judgement"
    if any(x in s for x in ["affidavit", "nori"]):
        return "affidavit"
    if any(x in s for x in ["legal notice", "notice"]):
        return "legal_notice"
    if any(x in s for x in ["agreement", "nda", "non disclosure", "vendor agreement"]):
        return "agreement"
    if any(x in s for x in ["contract", "indemnity", "sale of goods"]):
        return "contract"
    return "REVIEW_MANUALLY"

def main():
    files = []
    for folder in SEARCH_DIRS:
        if folder.exists():
            files.extend(p for p in folder.rglob("*.pdf") if p.is_file())

    seen_hashes = {}
    rows = []
    for p in sorted(set(files)):
        digest = sha256(p)
        duplicate_of = seen_hashes.get(digest, "") if digest else ""
        if digest and digest not in seen_hashes:
            seen_hashes[digest] = str(p.relative_to(ROOT))
        try:
            size = p.stat().st_size
        except OSError:
            size = 0

        try:
            relative = str(p.relative_to(ROOT))
        except ValueError:
            relative = str(p)

        rows.append({
            "file": p.name,
            "relative_path": relative,
            "folder": p.parent.name,
            "size_bytes": size,
            "sha256": digest,
            "duplicate_of": duplicate_of,
            "suggested_category_from_filename_only": suggested_category(p.name),
            "verified_category": "",
            "authenticity_checked": "NO",
            "notes": "Review PDF content and source before assigning final label",
        })

    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "relative_path", "folder", "size_bytes", "sha256",
            "duplicate_of", "suggested_category_from_filename_only",
            "verified_category", "authenticity_checked", "notes"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"PDF files found: {len(rows)}")
    print(f"Inventory saved: {OUTPUT}")
    print("Category suggestions are filename-only; review before training.")
    counts = {}
    for r in rows:
        counts[r["folder"]] = counts.get(r["folder"], 0) + 1
    for folder, count in sorted(counts.items()):
        print(f"  {folder}: {count}")

if __name__ == "__main__":
    main()
