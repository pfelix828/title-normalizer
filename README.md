# Job Title Normalizer

A multi-task PyTorch classifier that normalizes messy CRM job titles into standardized **seniority levels** and **job functions**. Built to solve a real problem in B2B marketing: contact databases contain thousands of title variations that need to be standardized for segmentation, targeting, and buying group analysis.

## The Problem

CRM and marketing databases are full of inconsistent job titles:

| Raw Title | Seniority | Function |
|-----------|-----------|----------|
| `Sr. Dir. of Eng` | senior_director | engineering |
| `VP Marketing & Growth` | vp | marketing |
| `senior data scientist II` | senior | data |
| `Mgr, Product` | manager | product |
| `CHIEF REVENUE OFFICER` | c_suite | sales |

Manual normalization doesn't scale. Rule-based approaches break on edge cases. This project trains a neural network to classify titles into 10 seniority levels and 10 job functions simultaneously.

## Results

| Model | Seniority Acc | Function Acc | Both Correct |
|-------|:---:|:---:|:---:|
| **CNN** (PyTorch) | 95.2% | 93.0% | 89.0% |
| **BiLSTM** (PyTorch) | 94.5% | 92.6% | 88.0% |
| TF-IDF + LogReg (baseline) | 93.3% | 93.3% | 87.6% |

*Trained on 16,000 synthetic titles, evaluated on 4,000 held-out. Synthetic data includes realistic CRM noise: truncation, misspellings, geo/team qualifiers, credential suffixes, encoding artifacts, word reordering, and separator variation.*

## Architecture

**Multi-task learning**: a single model predicts both seniority and function through shared representations.

```
Input (token IDs) → Embedding(64d) → BiLSTM(128d, bidirectional)
    → concat hidden states → Dropout(0.3)
        → Linear → Seniority (10 classes)
        → Linear → Function (10 classes)
```

Also includes a 1D CNN variant with multiple kernel sizes (2, 3, 4) for n-gram pattern matching.

### Classification Targets

**Seniority** (10 levels): individual_contributor, senior, lead, manager, senior_manager, director, senior_director, vp, svp, c_suite

**Function** (10 categories): engineering, data, product, marketing, sales, finance, hr, operations, design, legal

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic data
python -m src.generate_data

# Run baseline (TF-IDF + Logistic Regression)
python -m src.baseline

# Train PyTorch model
python -m src.train

# Run tests
pytest tests/ -v
```

## Project Structure

```
title-normalizer/
├── src/
│   ├── generate_data.py   # Synthetic title generation with realistic noise
│   ├── dataset.py         # PyTorch Dataset, Vocabulary, tokenization
│   ├── models.py          # BiLSTM and CNN classifiers
│   ├── baseline.py        # TF-IDF + Logistic Regression comparison
│   ├── train.py           # Training loop with early stopping
│   ├── evaluate.py        # Per-class precision, recall, F1
│   └── predict.py         # Single title and batch inference
├── tests/                 # 28 pytest tests
├── notebooks/             # Exploration and analysis
├── app/                   # Streamlit dashboard
├── data/                  # Generated CSVs (gitignored)
└── models/                # Saved model weights (gitignored)
```

## Synthetic Data

Generates realistic title variations from a taxonomy of 250+ canonical titles. Noise mirrors actual CRM data quality issues:

- **Abbreviations**: Senior → Sr., Director → Dir., Marketing → Mktg
- **Reordering**: "Director of Engineering" ↔ "Engineering, Director"
- **Level suffixes**: "Software Engineer II", "L5 Engineer", "IC4", "E6"
- **Case variation**: lowercase, UPPERCASE, Title Case, rAnDom MiXed
- **Filler words**: insertion/removal of "of", "&", "-", "/"
- **Misspellings**: real typos — "Manger", "Engineeer", "Sofware", "Marekting"
- **Whitespace noise**: extra spaces, tabs, inconsistent trimming
- **Truncation**: CRM field limits cutting titles mid-word — "Senior Vice Presi..."
- **Geo qualifiers**: "(EMEA)", "- North America", ", Global"
- **Team qualifiers**: "(Platform)", "- Payments", "(Growth)"
- **Credential suffixes**: ", MBA", ", CPA", ", PMP", " JD"
- **Separator variation**: "VP / Engineering", "Sales | Manager"
- **Word drops**: lazy data entry dropping words from titles
- **Encoding artifacts**: smart quotes, em dashes, HTML entities, non-breaking spaces

Noise level is configurable (0.0 = clean canonical titles, 1.0 = full variation).

## Methodology

1. **Baseline first**: TF-IDF (1-2 grams) + Logistic Regression establishes a performance floor
2. **Neural approach**: Word embeddings learned from scratch → BiLSTM captures sequential patterns → multi-task heads share representations between seniority and function prediction
3. **Early stopping**: Monitors validation loss with patience=5 to prevent overfitting
4. **Per-class evaluation**: Precision, recall, and F1 for every seniority level and function category

## Production Considerations

This project uses synthetic data to demonstrate the approach. For production deployment:

- **Real data**: Retrain on labeled titles from actual CRM exports (Salesforce, HubSpot)
- **Active learning**: Flag low-confidence predictions for human review
- **Title embedding**: Extract learned embeddings for downstream tasks (clustering, similarity search)
- **API serving**: Wrap the model in FastAPI for real-time classification
- **Monitoring**: Track prediction confidence distribution to detect data drift

## Tech Stack

Python, PyTorch, scikit-learn, Pandas, NumPy, pytest
