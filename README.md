# AI Legal Document Classification System

**BERT Classification | DistilBART Summarization | Legal Information Extraction**

## Project Overview

The AI Legal Document Classification System is a machine learning-based web application designed to automate the analysis of legal PDF documents.

The application uses a fine-tuned BERT model to classify legal documents into different categories and DistilBART to generate concise summaries. It also extracts important legal information to make document review easier and more efficient.

## Key Features

- **Automatic Document Classification:** Classifies uploaded PDF documents into:
  - Affidavit
  - Agreement
  - Contract
  - Court Judgement
  - Legal Notice

- **AI-Based Summarization:** Generates concise summaries using DistilBART.

- **Legal Information Extraction:** Identifies names, parties, important dates, financial amounts, deadlines and obligations.

- **Multiple PDF Upload:** Supports analysis of multiple legal documents.

- **Analytics Dashboard:** Displays document statistics, classification confidence and processing information.

- **PDF Analysis Report:** Generates downloadable reports containing document analysis results.

## Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Flask |
| Classification Model | BERT |
| Summarization Model | DistilBART |
| Deep Learning Framework | PyTorch |
| NLP Framework | Hugging Face Transformers |
| PDF Text Extraction | PyMuPDF |
| Report Generation | ReportLab |
| Frontend | HTML, CSS |

## System Workflow

1. User uploads one or more legal PDF documents.
2. Text is extracted from the uploaded PDFs.
3. The BERT model predicts the document category.
4. DistilBART generates a concise document summary.
5. Legal information is extracted from the document.
6. Results are displayed on the web dashboard.
7. Users can download PDF analysis reports.

## Project Structure

```text
AI-Legal-Document-Classification/
│
├── main.py
├── dataset/
├── models/
├── public_legal_source_download/
├── uploads/
├── templates/
├── static/
├── README.md
└── requirements.txt
```

*The actual repository structure may vary. Large trained model files, virtual environments and private uploaded documents should not be committed to GitHub.*

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd AI-Legal-Document-Classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```powershell
venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python main.py web
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

**Note:** The trained BERT classifier and required summarization model must be available locally for the application to run. Follow the project's model setup instructions.

## Institution Details

### Sai Vidya Institute of Technology (SVIT)

Sai Vidya Institute of Technology (SVIT) is an engineering institution located at Rajanukunte, Yelahanka, Bengaluru, Karnataka, India.

Established in 2008 under Sri Sai Vidya Vikas Shikshana Samithi, the institution focuses on technical education, innovation, research and the overall development of students.

SVIT is an autonomous institution under Visvesvaraya Technological University (VTU), Belagavi, and is approved by the All India Council for Technical Education (AICTE).

**Institution:** Sai Vidya Institute of Technology  
**Location:** Rajanukunte, Bengaluru, Karnataka  
**Website:** https://www.saividya.ac.in/

## Academic Project

This project is developed as part of an academic learning initiative in Artificial Intelligence and Machine Learning, applying natural language processing and deep learning techniques to automate legal document analysis.

## Disclaimer

This application is intended for educational and research purposes. AI-generated classifications, summaries and extracted information may contain errors and should not be treated as legal advice. Results must be reviewed by a qualified legal professional.

## Acknowledgement

We sincerely acknowledge Sai Vidya Institute of Technology for providing the academic environment and learning opportunities that support innovation and technical project development.
