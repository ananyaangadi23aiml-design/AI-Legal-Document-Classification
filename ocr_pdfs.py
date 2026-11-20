
from pathlib import Path
import pymupdf
import pytesseract
from PIL import Image
import io
import csv

root = Path("public_legal_source_download")
names = [
    "arbitration clause 2.pdf",
    "arbitration clause.pdf",
    "NDA template.pdf",
    "wrongful termination employment case 2.pdf",
    "wrongful termination employment.pdf",
]

# If Tesseract is not found automatically, uncomment the next line
# and change the path to your actual tesseract.exe location.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

results = []

for name in names:
    pdf_path = next(root.rglob(name), None)
    if pdf_path is None:
        print(f"NOT FOUND: {name}")
        continue

    extracted = []
    try:
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                extracted.append(pytesseract.image_to_string(image))

        text = "\n".join(extracted)
        results.append({
            "filename": name,
            "pages": len(extracted),
            "characters": len(text),
            "text_preview": " ".join(text.split())[:1000],
            "ocr_text": text,
        })
        print(f"{name}: OCR extracted {len(text)} characters")

    except Exception as e:
        print(f"OCR ERROR for {name}: {e}")

if results:
    with open("ocr_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Saved OCR results to ocr_results.csv")
