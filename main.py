import os
import re
import json
import uuid
import time
import html
import pymupdf
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from xml.sax.saxutils import escape
from flask import (
    Flask, request, render_template_string,
    send_file, abort
)
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline
)
# PDF report generation
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models" / "legal_bert"
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DATA_DIR = REPORT_DIR / "report_data"
ANALYTICS_FILE = BASE_DIR / "analytics.json"
for folder in [
    MODEL_DIR,
    UPLOAD_DIR,
    REPORT_DIR,
    REPORT_DATA_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)
CATEGORIES = [
    "affidavit",
    "agreement",
    "contract",
    "court_judgement",
    "legal_notice"
]
LABEL2ID = {
    label: i for i, label in enumerate(CATEGORIES)
}
ID2LABEL = {
    i: label for i, label in enumerate(CATEGORIES)
}
BASE_MODEL = "bert-base-uncased"
SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
DEVICE = torch.device("cpu")
# Limit CPU thread contention during inference.
CPU_THREADS = max(1, min(4, os.cpu_count() or 1))
torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
# Cache models instead of loading them for every document
classifier_model = None
summary_tokenizer = None
summary_model = None
# ============================================================
# MODEL LOADING
# ============================================================
def load_classifier():
    global classifier_model
    if classifier_model is not None:
        return classifier_model
    classifier_dir = MODEL_DIR / "classifier"
    if not (classifier_dir / "config.json").exists():
        print("Trained BERT model not found.")
        print("Run: python main.py train")
        return None
    print("Loading trained BERT classifier...")
    tokenizer = AutoTokenizer.from_pretrained(
        classifier_dir
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        classifier_dir
    )
    classifier_model = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )
    print("BERT classifier loaded.")
    return classifier_model
def load_summarizer():
    global summary_tokenizer, summary_model

    if summary_model is not None:
        return summary_tokenizer, summary_model

    print("Loading DistilBART summarization model...")

    summary_tokenizer = AutoTokenizer.from_pretrained(
        SUMMARY_MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARY_MODEL
    )

    model.to(DEVICE)
    model.eval()

    summary_model = model

    print("DistilBART summarization model loaded in FP32 mode.")

    return summary_tokenizer, summary_model
# PDF TEXT EXTRACTION
# ============================================================
def extract_text_from_pdf(pdf_path):
    document = pymupdf.open(pdf_path)
    try:
        page_texts = [page.get_text("text") for page in document]
        return "\n".join(page_texts)
    finally:
        document.close()
# TEXT PREPROCESSING
# ============================================================
def preprocess_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
# ============================================================
# DATASET
# ============================================================
def load_dataset():
    texts = []
    labels = []
    for category in CATEGORIES:
        category_dir = DATASET_DIR / category
        if not category_dir.exists():
            print(f"Missing folder: {category_dir}")
            continue
        pdf_files = list(category_dir.glob("*.pdf"))
        print(f"{category}: {len(pdf_files)} PDF files")
        for pdf_file in pdf_files:
            try:
                text = extract_text_from_pdf(pdf_file)
                text = preprocess_text(text)
                if len(text) > 50:
                    texts.append(text)
                    labels.append(LABEL2ID[category])
            except Exception as e:
                print(f"Error reading {pdf_file}: {e}")
    return texts, labels
# ============================================================
# TOKENIZATION DATASET
# ============================================================
class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index]
        )
        return item
# ============================================================
# TRAIN BERT MODEL
# ============================================================
def train_model():
    print("\nLoading dataset...")
    texts, labels = load_dataset()
    if len(texts) < 10:
        print("Not enough documents.")
        return
    train_texts, test_texts, train_labels, test_labels = (
        train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
    )
    print("\nTraining documents:", len(train_texts))
    print("Testing documents:", len(test_texts))
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )
    train_dataset = LegalDataset(
        train_texts,
        train_labels,
        tokenizer
    )
    test_dataset = LegalDataset(
        test_texts,
        test_labels,
        tokenizer
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )
    epochs = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(train_loader):
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (step + 1) % 10 == 0:
                print(
                    f"Batch {step + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )
        print(
            f"Epoch {epoch + 1} Average Loss: "
            f"{total_loss / len(train_loader):.4f}"
        )
    print("\nEvaluating model...")
    model.eval()
    predicted_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }
            batch.pop("labels")
            outputs = model(**batch)
            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )
            predicted_labels.extend(
                predictions.cpu().numpy()
            )
    print("\nClassification Report:\n")
    print(classification_report(
        test_labels,
        predicted_labels,
        labels=list(range(len(CATEGORIES))),
        target_names=CATEGORIES,
        zero_division=0
    ))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(
        test_labels,
        predicted_labels,
        labels=list(range(len(CATEGORIES)))
    ))
    classifier_dir = MODEL_DIR / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(classifier_dir))
    tokenizer.save_pretrained(str(classifier_dir))
    print("\nTraining completed successfully.")
    print("Model saved at:", classifier_dir)
# ============================================================
# DOCUMENT CLASSIFICATION
# ============================================================
def classify_document(text):
    classifier = load_classifier()
    if classifier is None:
        return None
    result = classifier(
        text[:5000],
        truncation=True
    )[0]
    label = result["label"].lower()
    confidence = result["score"]
    return label, confidence
# ============================================================
# DOCUMENT SUMMARIZATION
# ============================================================
def summarize_document(text):
    try:
        tokenizer, model = load_summarizer()

        if len(text) < 100:
            return text

        inputs = tokenizer(
            text[:6000],
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        with torch.inference_mode():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=120,
                min_length=25,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )

        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary

    except Exception as e:
        print("Summarization error:", e)
        return f"Summary generation failed: {e}"
# FEATURE 2: IMPORTANT LEGAL INFORMATION EXTRACTOR
# ============================================================
def extract_legal_information(text):
    import re
    if not text:
        return {
            "names": [],
            "dates": [],
            "amounts": [],
            "deadlines": [],
            "obligations": []
        }
    # Normalize extracted PDF text
    clean_text = re.sub(r"\s+", " ", text).strip()
    # --------------------------------------------------
    # 1. NAME AND PARTY EXTRACTION
    # --------------------------------------------------
    names = set()
    # Matches normal person names, including minor
    # capitalization errors such as MehtA.
    person_pattern = (
        r"[A-Z][a-zA-Z'-]+"
        r"(?:\s+[A-Z][a-zA-Z'-]+){1,3}"
    )
    def clean_person_name(name):
        name = name.strip(" ,.;:-")
        # Remove common legal phrases accidentally captured
        name = re.split(
            r"\b(?:solemnly|affirmed|deponent|party|"
            r"aged|residing|having|both|and)\b",
            name,
            flags=re.IGNORECASE
        )[0].strip()
        words = name.split()
        if len(words) < 2 or len(words) > 4:
            return None
        # Normalize capitalization
        words = [word.capitalize() for word in words]
        # Reject common non-person labels
        invalid = {
            "This Agreement",
            "The Parties",
            "Competent Authority",
            "Concerned Authority"
        }
        result = " ".join(words)
        if result in invalid:
            return None
        return result
    def add_person(match):
        candidate = clean_person_name(match)
        if candidate:
            names.add(candidate)
    # Affidavit: I, Arjun Mehta
    for match in re.findall(
        r"\bI\s*,\s*(" + person_pattern + r")",
        clean_text
    ):
        add_person(match)
    # Agreement: between Arjun Mehta and Priya Mehta
    between_match = re.search(
        r"\bbetween\s+(" + person_pattern +
        r")\s+and\s+(" + person_pattern + r")",
        clean_text,
        re.IGNORECASE
    )
    if between_match:
        add_person(between_match.group(1))
        add_person(between_match.group(2))
    # Party A / Party B labels
    for match in re.findall(
        r"\bParty\s*[AB]\s*:\s*(" + person_pattern + r")",
        clean_text,
        re.IGNORECASE
    ):
        add_person(match)
    # Deponent / Applicant / Respondent / etc.
    role_pattern = (
        r"\b(?:DEPONENT|Applicant|Respondent|"
        r"Plaintiff|Defendant|Witness|"
        r"Borrower|Lender|Seller|Buyer)"
        r"\s*:\s*(" + person_pattern + r")"
    )
    for match in re.findall(
        role_pattern,
        clean_text,
        re.IGNORECASE
    ):
        add_person(match)
    # Labels such as Name: Arjun Mehta
    for match in re.findall(
        r"\bName\s*:\s*(" + person_pattern + r")",
        clean_text,
        re.IGNORECASE
    ):
        add_person(match)
    # --------------------------------------------------
    # 2. DATE EXTRACTION
    # --------------------------------------------------
    date_pattern = (
        r"\b(?:"
        r"\d{1,2}\s+(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)\s+\d{4}"
        r"|"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4}"
        r"|"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r")\b"
    )
    dates = list(dict.fromkeys(
        re.findall(date_pattern, clean_text, re.IGNORECASE)
    ))
    # --------------------------------------------------
    # 3. FINANCIAL AMOUNT EXTRACTION
    # --------------------------------------------------
    amount_pattern = (
        r"(?:₹|Rs\.?|INR|\$|USD)\s*"
        r"\d[\d,]*(?:\.\d{1,2})?"
        r"|"
        r"\b\d[\d,]*(?:\.\d{1,2})?\s*"
        r"(?:rupees|INR|USD|dollars)\b"
    )
    amounts = list(dict.fromkeys(
        re.findall(amount_pattern, clean_text, re.IGNORECASE)
    ))
    # --------------------------------------------------
    # 4. DEADLINE EXTRACTION
    # --------------------------------------------------
    deadline_keywords = [
        "within",
        "no later than",
        "on or before",
        "deadline",
        "due date",
        "before the expiry",
        "by the date"
    ]
    sentences = re.split(
        r"(?<=[.!?;])\s+",

        clean_text
    )
    deadlines = []
    for sentence in sentences:
        if any(
            keyword in sentence.lower()
            for keyword in deadline_keywords
        ):
            sentence = sentence.strip()
            if sentence and sentence not in deadlines:
                deadlines.append(sentence)
    # --------------------------------------------------
    # 5. IMPORTANT OBLIGATIONS
    # --------------------------------------------------
    obligation_keywords = [
        "shall",
        "must",
        "undertakes to",
        "is required to",
        "agrees to",
        "will be responsible",
        "shall not",
        "is obligated to"
    ]
    obligations = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if any(
            keyword in sentence.lower()
            for keyword in obligation_keywords
        ):
            # Avoid excessively long extracted paragraphs
            if len(sentence) > 350:
                parts = re.split(
                    r"(?<=[.!?])\s+|;\s+",
                    sentence
                )
                for part in parts:
                    part = part.strip()
                    if (
                        part
                        and any(
                            keyword in part.lower()
                            for keyword in obligation_keywords
                        )
                        and len(part) <= 350
                    ):
                        if part not in obligations:
                            obligations.append(part)
            else:
                if sentence not in obligations:
                    obligations.append(sentence)
    # --------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------
    return {
        "names": sorted(names),
        "dates": dates,
        "amounts": amounts,
        "deadlines": deadlines,
        "obligations": obligations
    }
# ============================================================
# FEATURE 3: ANALYTICS DASHBOARD
# ============================================================
def load_analytics():
    if not ANALYTICS_FILE.exists():
        return {
            "documents": []
        }
    try:
        with open(
            ANALYTICS_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)
        if "documents" not in data:
            return {"documents": []}
        return data
    except (json.JSONDecodeError, OSError):
        return {
            "documents": []
        }
def save_analytics(data):
    temp_file = ANALYTICS_FILE.with_suffix(".tmp")
    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=4
        )
    os.replace(
        temp_file,
        ANALYTICS_FILE
    )
def update_analytics(result):
    data = load_analytics()
    data["documents"].append({
        "filename": result["filename"],
        "category": result["category"],
        "confidence": result["confidence"],
        "processing_time": result["processing_time"],
        "analysis_date": result["analysis_date"]
    })
    save_analytics(data)
def get_dashboard_statistics():
    data = load_analytics()
    documents = data.get("documents", [])
    total = len(documents)
    category_counts = {
        category: 0
        for category in CATEGORIES
    }
    for document in documents:
        category = document.get("category")
        if category in category_counts:
            category_counts[category] += 1
    if total:
        average_confidence = round(
            sum(
                doc["confidence"]
                for doc in documents
            ) / total,
            2
        )
        average_processing_time = round(
            sum(
                doc["processing_time"]
                for doc in documents
            ) / total,
            2
        )
    else:
        average_confidence = 0
        average_processing_time = 0
    recent_documents = list(
        reversed(documents[-10:])
    )
    return {
        "total": total,
        "category_counts": category_counts,
        "average_confidence": average_confidence,
        "average_processing_time": average_processing_time,
        "recent_documents": recent_documents
    }
# ============================================================
# FEATURE 4: REPORT DATA STORAGE
# ============================================================
def save_report_data(result):
    report_id = str(uuid.uuid4())
    report_file = REPORT_DATA_DIR / f"{report_id}.json"
    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )
    return report_id
# ============================================================
# FEATURE 4: PDF REPORT GENERATION
# ============================================================
def create_pdf_report(result):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#172554"),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        spaceAfter=6
    )
    story = []
    story.append(
        Paragraph(
            "AI LEGAL DOCUMENT ANALYSIS REPORT",
            title_style
        )
    )
    story.append(
        Paragraph(
            "Automated document classification and analysis",
            body_style
        )
    )
    story.append(Spacer(1, 12))
    metadata = [
        ["Original Filename", result["filename"]],
        ["Predicted Category", result["category"]],
        ["Classification Confidence", f'{result["confidence"]}%'],
        ["Analysis Date", result["analysis_date"]],
        ["Processing Time", f'{result["processing_time"]} seconds'],
        ["Extracted Text Length", f'{result["text_length"]} characters']
    ]
    table = Table(
        metadata,
        colWidths=[55 * mm, 110 * mm]
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
    ]))
    story.append(table)
    story.append(
        Paragraph("Document Summary", heading_style)
    )
    story.append(
        Paragraph(
            escape(result["summary"]),
            body_style
        )
    )
    story.append(
        Paragraph("Important Legal Information", heading_style)
    )
    information = result["legal_information"]
    sections = [
        ("Names and Parties", "names"),
        ("Important Dates", "dates"),
        ("Financial Amounts", "amounts"),
        ("Deadlines", "deadlines"),
        ("Important Obligations", "obligations")
    ]
    for heading, key in sections:
        story.append(
            Paragraph(heading, styles["Heading3"])
        )
        items = information.get(key, [])
        if items:
            for item in items:
                story.append(
                    Paragraph(
                        "• " + escape(str(item)),
                        body_style
                    )
                )
        else:
            story.append(
                Paragraph(
                    "No information detected.",
                    body_style
                )
            )
    story.append(
        Paragraph("Complete Extracted Text", heading_style)
    )
    extracted_text = result.get("extracted_text", "")
    # Split long text into smaller paragraphs for PDF layout
    for start in range(0, len(extracted_text), 1500):
        chunk = extracted_text[start:start + 1500]
        story.append(
            Paragraph(
                escape(chunk),
                body_style
            )
        )
    story.append(Spacer(1, 15))
    story.append(
        Paragraph(
            "Disclaimer: This report is generated automatically "
            "using machine learning and rule-based extraction. "
            "Extracted information may be incomplete or inaccurate. "
            "It is not a substitute for professional legal review.",
            styles["Italic"]
        )
    )
    document.build(story)
    buffer.seek(0)
    return buffer
# ============================================================
# PROCESS PDF
# ============================================================
def process_pdf(pdf_path, original_filename):
    start_time = time.time()
    try:
        text = extract_text_from_pdf(pdf_path)
        text = preprocess_text(text)
        if not text:
            return {
                "error": "No text could be extracted from this PDF."
            }
        classification = classify_document(text)
        if classification is None:
            return {
                "error": "BERT model is not trained yet."
            }
        category, confidence = classification
        summary = summarize_document(text)
        legal_information = extract_legal_information(text)
        processing_time = round(
            time.time() - start_time,
            2
        )
        result = {
            "filename": original_filename,
            "category": category,
            "confidence": round(confidence * 100, 2),
            "summary": summary,
            "text_length": len(text),
            "extracted_text": text,
            "legal_information": legal_information,
            "processing_time": processing_time,
            "analysis_date": datetime.now().strftime(
                "%d-%m-%Y %I:%M:%S %p"
            )
        }
        report_id = save_report_data(result)
        result["report_id"] = report_id
        update_analytics(result)
        return result
    except Exception as e:
        print(f"Processing error: {e}")
        return {
            "error": str(e)
        }
# ============================================================
# WEB APPLICATION DESIGN
# ============================================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>AI Legal Document Classification System</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {
    box-sizing: border-box;
}
body {
    font-family: Arial, sans-serif;
    background: #f1f5f9;
    margin: 0;
    padding: 25px;
    color: #1e293b;
}
.container {
    max-width: 1150px;
    margin: auto;
}
.header {
    background: #172554;
    color: white;
    padding: 28px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 25px;
}
.header h1 {
    margin: 0 0 10px;
}
.header p {
    margin: 0;
    color: #cbd5e1;
}
.section {
    background: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.section h2 {
    margin-top: 0;
    color: #172554;
}
.upload-box {
    border: 2px dashed #94a3b8;
    border-radius: 10px;
    padding: 30px;
    text-align: center;
    background: #f8fafc;
}
button, .download-btn {
    background: #172554;
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 7px;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
    font-size: 14px;
    margin-top: 12px;
}
button:hover, .download-btn:hover {
    background: #1e40af;
}
button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
.file-list {
    margin-top: 15px;
    text-align: left;
}
.file-item {
    background: white;
    padding: 10px;
    margin: 6px 0;
    border-radius: 6px;
    display: flex;
    justify-content: space-between;
    gap: 10px;
    overflow-wrap: anywhere;
}
.remove-file {
    color: #dc2626;
    cursor: pointer;
    font-weight: bold;
}
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 15px;
}
.stat-card {
    padding: 20px;
    background: #eff6ff;
    border-radius: 10px;
    border-left: 4px solid #2563eb;
}
.stat-card h3 {
    margin: 0 0 10px;
    font-size: 14px;
    color: #475569;
}
.stat-card .number {
    font-size: 28px;
    font-weight: bold;
    color: #172554;
}
.category-row {
    margin: 15px 0;
}
.category-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    font-size: 14px;
}
.progress {
    height: 12px;
    background: #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    background: #2563eb;
    border-radius: 10px;
}
.result {
    background: #f8fafc;
    padding: 22px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 1px solid #e2e8f0;
}
.result h3 {
    color: #172554;
}
.category {
    font-size: 23px;
    font-weight: bold;
    color: #1d4ed8;
    text-transform: capitalize;
}
.summary {
    line-height: 1.7;
}
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
}
.info-card {
    background: white;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}
.info-card h4 {
    margin-top: 0;
    color: #1e40af;
}
.info-card li {
    margin-bottom: 7px;
    line-height: 1.4;
    overflow-wrap: anywhere;
}
details {
    margin-top: 15px;
}
pre {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    background: white;
    padding: 15px;
    border-radius: 8px;
    max-height: 350px;
    overflow-y: auto;
    line-height: 1.5;
}
table {
    width: 100%;
    border-collapse: collapse;
}
th, td {
    text-align: left;
    padding: 11px;
    border-bottom: 1px solid #e2e8f0;
    font-size: 14px;
}
th {
    background: #f8fafc;
}
.error {
    background: #fee2e2;
    color: #991b1b;
    padding: 14px;
    border-radius: 8px;
    margin-bottom: 15px;
}
.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
    font-size: 13px;
}
@media(max-width: 600px) {
    body {
        padding: 12px;
    }
    .section {
        padding: 16px;
    }
}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>AI Legal Document Classification System</h1>
<p>
BERT Classification | DistilBART Summarization |
Legal Information Extraction
</p>
</div>
<!-- ANALYTICS DASHBOARD -->
<div class="section">
<h2>Document Analytics Dashboard</h2>
<div class="cards">
<div class="stat-card">
<h3>Total Documents Processed</h3>
<div class="number">{{ stats.total }}</div>
</div>
<div class="stat-card">
<h3>Average Confidence</h3>
<div class="number">{{ stats.average_confidence }}%</div>
</div>
<div class="stat-card">
<h3>Average Processing Time</h3>
<div class="number">{{ stats.average_processing_time }}s</div>
</div>
</div>
<h3>Category-wise Distribution</h3>
{% for category, count in stats.category_counts.items() %}
<div class="category-row">
<div class="category-label">
<span>{{ category.replace('_', ' ').title() }}</span>
<strong>{{ count }}</strong>
</div>
<div class="progress">
<div class="progress-fill"
style="width: {{ (count / stats.total * 100) if stats.total else 0 }}%">
</div>
</div>
</div>
{% endfor %}
</div>
<!-- UPLOAD SECTION -->
<div class="section">
<h2>Upload Legal Documents</h2>
<div class="upload-box">
<form method="POST"
action="/predict"
enctype="multipart/form-data"
id="uploadForm">
<input
type="file"
id="fileInput"
name="files"
accept=".pdf,application/pdf"
multiple
>
<div id="fileList" class="file-list"></div>
<br>
<button type="submit" id="submitButton">
Analyze Selected Documents
</button>
</form>
</div>
</div>
<!-- ANALYSIS RESULTS -->
{% if results %}
<div class="section">
<h2>Document Analysis Results</h2>
{% for result in results %}
{% if result.error %}
<div class="error">
<strong>{{ result.filename }}:</strong>
{{ result.error }}
</div>
{% else %}
<div class="result">
<h3>{{ result.filename }}</h3>
<p><strong>Predicted Category</strong></p>
<div class="category">
{{ result.category.replace('_', ' ') }}
</div>
<p>
<strong>Classification Confidence:</strong>
{{ result.confidence }}%
</p>
<p>
<strong>Extracted Text Length:</strong>
{{ result.text_length }} characters
</p>
<p>
<strong>Processing Time:</strong>
{{ result.processing_time }} seconds
</p>
<p>
<strong>Analysis Date:</strong>
{{ result.analysis_date }}
</p>
<h3>Document Summary</h3>
<p class="summary">
{{ result.summary }}
</p>
<h3>Important Legal Information</h3>
<div class="info-grid">
{% set info = result.legal_information %}
<div class="info-card">
<h4>Names and Parties</h4>
{% if info.names %}
<ul>
{% for item in info.names %}
<li>{{ item }}</li>
{% endfor %}
</ul>
{% else %}
<p>No information detected.</p>
{% endif %}
</div>
<div class="info-card">
<h4>Important Dates</h4>
{% if info.dates %}
<ul>
{% for item in info.dates %}
<li>{{ item }}</li>
{% endfor %}
</ul>
{% else %}
<p>No information detected.</p>
{% endif %}
</div>
<div class="info-card">
<h4>Financial Amounts</h4>
{% if info.amounts %}
<ul>
{% for item in info.amounts %}
<li>{{ item }}</li>
{% endfor %}
</ul>
{% else %}
<p>No information detected.</p>
{% endif %}
</div>
<div class="info-card">
<h4>Deadlines</h4>
{% if info.deadlines %}
<ul>
{% for item in info.deadlines %}
<li>{{ item }}</li>
{% endfor %}
</ul>
{% else %}
<p>No information detected.</p>
{% endif %}
</div>
<div class="info-card">
<h4>Important Obligations</h4>
{% if info.obligations %}
<ul>
{% for item in info.obligations %}
<li>{{ item }}</li>
{% endfor %}
</ul>
{% else %}
<p>No information detected.</p>
{% endif %}
</div>
</div>
<details>
<summary><strong>View Complete Extracted Text</strong></summary>
<pre>{{ result.extracted_text }}</pre>
</details>
<a class="download-btn"
href="/report/{{ result.report_id }}">
Download PDF Analysis Report
</a>
</div>
{% endif %}
{% endfor %}
</div>
{% endif %}
<!-- RECENT ANALYSIS -->
<div class="section">
<h2>Recently Processed Documents</h2>
{% if stats.recent_documents %}
<div style="overflow-x:auto;">
<table>
<thead>
<tr>
<th>Filename</th>
<th>Category</th>
<th>Confidence</th>
<th>Processing Time</th>
</tr>
</thead>
<tbody>
{% for doc in stats.recent_documents %}
<tr>
<td>{{ doc.filename }}</td>
<td>{{ doc.category }}</td>
<td>{{ doc.confidence }}%</td>
<td>{{ doc.processing_time }}s</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>
{% else %}
<p>No documents have been processed yet.</p>
{% endif %}
</div>
<div class="footer">
AI Legal Document Classification System
<br>
Automated results should be reviewed by a qualified legal professional.
</div>
</div>
<script>
const fileInput = document.getElementById("fileInput");
const fileList = document.getElementById("fileList");
const uploadForm = document.getElementById("uploadForm");
const submitButton = document.getElementById("submitButton");
let selectedFiles = [];
function updateFileList() {
    fileList.innerHTML = "";
    selectedFiles.forEach((file, index) => {
        const item = document.createElement("div");
        item.className = "file-item";
        const name = document.createElement("span");
        name.textContent = file.name;
        const remove = document.createElement("span");
        remove.className = "remove-file";
        remove.textContent = "Remove";
        remove.addEventListener("click", () => {
            selectedFiles.splice(index, 1);
            updateFileList();
        });
        item.appendChild(name);
        item.appendChild(remove);
        fileList.appendChild(item);
    });
    submitButton.disabled = selectedFiles.length === 0;
    if (selectedFiles.length > 0) {
        submitButton.textContent =
            "Analyze " + selectedFiles.length + " Document(s)";
    } else {
        submitButton.textContent = "Analyze Selected Documents";
    }
}
fileInput.addEventListener("change", function() {
    const newlySelectedFiles = Array.from(this.files);
    newlySelectedFiles.forEach(file => {
        if (
            file.type === "application/pdf" ||
            file.name.toLowerCase().endsWith(".pdf")
        ) {
            const duplicate = selectedFiles.some(
                existing =>
                    existing.name === file.name &&
                    existing.size === file.size &&
                    existing.lastModified === file.lastModified
            );
            if (!duplicate) {
                selectedFiles.push(file);
            }
        }
    });
    fileInput.value = "";
    updateFileList();
});
uploadForm.addEventListener("submit", function(event) {
    if (selectedFiles.length === 0) {
        event.preventDefault();
        alert("Please select at least one PDF.");
        return;
    }
    const dataTransfer = new DataTransfer();
    selectedFiles.forEach(file => {
        dataTransfer.items.add(file);
    });
    fileInput.files = dataTransfer.files;
    submitButton.disabled = true;
    submitButton.textContent = "Processing Documents...";
});
</script>
</body>
</html>
"""
# ============================================================
# HOME
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_PAGE,
        results=None,
        stats=get_dashboard_statistics()
    )
# ============================================================
# MULTI-PDF UPLOAD AND ANALYSIS
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("files")
    if not files:
        return render_template_string(
            HTML_PAGE,
            results=[{
                "filename": "Upload",
                "error": "No files uploaded."
            }],
            stats=get_dashboard_statistics()
        )
    results = []
    for file in files:
        if not file or not file.filename:
            continue
        original_filename = secure_filename(
            file.filename
        )
        if not original_filename.lower().endswith(".pdf"):
            results.append({
                "filename": original_filename,
                "error": "Please upload a PDF file."
            })
            continue
        unique_filename = (
            str(uuid.uuid4()) + "_" + original_filename
        )
        file_path = UPLOAD_DIR / unique_filename
        try:
            file.save(file_path)
            result = process_pdf(
                file_path,
                original_filename
            )
            if result.get("error"):
                result["filename"] = original_filename
            results.append(result)
        except Exception as e:
            results.append({
                "filename": original_filename,
                "error": str(e)
            })
        finally:
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass
    return render_template_string(
        HTML_PAGE,
        results=results,
        stats=get_dashboard_statistics()
    )
# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================
@app.route("/report/<report_id>", methods=["GET"])
def download_report(report_id):
    if not re.fullmatch(
        r"[0-9a-fA-F-]{36}",
        report_id
    ):
        abort(404)
    report_file = REPORT_DATA_DIR / f"{report_id}.json"
    if not report_file.exists():
        abort(404)
    try:
        with open(
            report_file,
            "r",
            encoding="utf-8"
        ) as file:
            result = json.load(file)
        pdf_buffer = create_pdf_report(result)
        filename = secure_filename(
            Path(result["filename"]).stem
        )
        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{filename}_analysis_report.pdf"
        )
    except Exception as e:
        print("Report generation error:", e)
        return f"Unable to generate report: {e}", 500
# ============================================================
# COMMAND LINE
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "train":
            train_model()
        elif command == "predict":
            if len(sys.argv) < 3:
                print(
                    "Usage: python main.py predict file.pdf"
                )
            else:
                pdf_path = sys.argv[2]
                result = process_pdf(
                    pdf_path,
                    Path(pdf_path).name
                )
                print(
                    json.dumps(
                        result,
                        indent=4,
                        ensure_ascii=False
                    )
                )
        elif command == "web":
            print("Initializing AI models once at startup...")
            load_classifier()
            load_summarizer()
            classifier = load_classifier()
            if classifier is not None:
                try:
                    classifier("This is a legal document.", truncation=True)
                except Exception as warmup_error:
                    print("BERT warm-up skipped:", warmup_error)
            try:
                summarize_document(
                    "This is a short legal document used to initialize "
                    "the summarization model before document processing."
                )
            except Exception as warmup_error:
                print("Summarizer warm-up skipped:", warmup_error)
            print("AI models are ready. Starting Flask.")
            app.run(
                host="127.0.0.1",
                port=5000,
                debug=False,
                threaded=True
            )
        else:
            print("Commands: train, predict, web")
    else:
        print("Commands: train, predict, web")