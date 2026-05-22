# Raw Dataset Files

Place the original Kaggle dataset files here without editing them.

The initial fraud-detection workflow expects:

- `transactions_data.csv`
- `train_fraud_labels.json`

Keep the other downloaded files here too when you need user, card, and merchant
context for EDA or feature engineering:

- `users_data.csv`
- `cards_data.csv`
- `mcc_codes.json`

Run `notebooks/01_data_audit.ipynb` after the files are present.
