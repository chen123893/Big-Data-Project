# Big Data Fraud Detection Project

This project uses Python and Jupyter notebooks to analyse a financial transaction
fraud dataset, prepare cleaned data, compare machine learning models, and build a
small dashboard or prototype.

## First Steps

1. Download the Kaggle dataset into `data/raw/`.
2. Run `notebooks/01_data_audit.ipynb` before cleaning or modelling.
3. Record the audit results in the report: dataset source, file schema, missing
   values, duplicates, label balance, data-quality issues, privacy choices, and
   cleaning decisions.
4. Use `src/01_process_data.py` or a later preprocessing module to create
   cleaned files in `data/processed/`.
5. Build EDA, two machine learning models, evaluation plots, and a dashboard.

## Suggested Folder Contents

| Path | Purpose |
| --- | --- |
| `data/raw/` | Original downloaded Kaggle files. Do not edit them. |
| `data/processed/` | Cleaned and feature-ready outputs produced by code. |
| `notebooks/` | Jupyter notebooks for audit, EDA, modelling experiments, and result explanation. |
| `src/` | Reusable Python scripts for cleaning, features, training, evaluation, and utilities. |
| `models/` | Saved trained model artifacts created locally. |
| `app/` | Streamlit dashboard or prototype code. |
| `reports/` | Report drafts, exported tables, diagrams, and figure notes. |
| `reports/figures/` | Final charts and diagrams used in the report or slides. |
| `tests/` | Small checks for reusable preprocessing and modelling functions. |

## Recommended Notebook Order

1. `01_data_audit.ipynb`: confirm files, schema, missingness, duplicates,
   invalid parsing, label balance, and privacy-sensitive columns.
2. `02_eda.ipynb`: descriptive statistics and visualisations.
3. `03_preprocessing_features.ipynb`: document cleaning and feature choices.
4. `04_models.ipynb`: compare at least two algorithms.
5. `05_results.ipynb`: evaluation plots, limitations, and report-ready outputs.

## Project Notes

- Keep raw files unchanged so the cleaning pipeline is reproducible.
- Fraud detection is usually class-imbalanced, so report precision, recall,
  F1-score, confusion matrices, and false-positive trade-offs instead of relying
  on accuracy alone.
- Add commits as work progresses. The assignment expects version-control
  evidence, a clear README, and explainable code files.
