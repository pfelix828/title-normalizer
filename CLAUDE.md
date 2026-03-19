# Title Normalizer — Developer Instructions

## Quick Start
```bash
python -m src.generate_data          # Generate synthetic data
python -m src.baseline               # Run TF-IDF baseline
python -m src.train                  # Train PyTorch model
pytest tests/ -v                     # Run tests
streamlit run app/streamlit_app.py   # Launch dashboard
```

## Architecture
- `src/generate_data.py` — Synthetic title generation with noise/variations
- `src/dataset.py` — PyTorch Dataset, Vocabulary, tokenization
- `src/models.py` — BiLSTM and CNN classifiers (multi-task: seniority + function)
- `src/baseline.py` — TF-IDF + Logistic Regression baseline
- `src/train.py` — Training loop with early stopping
- `src/evaluate.py` — Per-class metrics and reporting
- `src/predict.py` — Single title and batch inference

## Key Design Decisions
- Multi-task learning: single model predicts both seniority and function
- Synthetic data mirrors real CRM noise (abbreviations, typos, reordering)
- Baseline comparison validates neural network adds value over classical ML
