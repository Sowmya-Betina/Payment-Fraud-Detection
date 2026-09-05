# Fraud Detection with ResNeXt-GRU + Attention (Jaya-tuned)

Online Payment Fraud detection on two independent datasets (IEEE-CIS and
PaySim), comparing classic baselines (Logistic Regression, XGBoost),
ResNeXt-GRU architecture, an added attention variant,
and a version of this model tuned with the Jaya optimisation algorithm
("RXT-J").

## Datasets

- **IEEE-CIS Fraud Detection** (Kaggle) — 590,540 transactions, ~3.5% fraud.
- **PaySim** (Kaggle, `ealaxi/paysim1`) — simulated mobile-money transactions,
  ~0.13% fraud, used as a second dataset to check the model generalisation
  beyond IEEE-CIS.

## Notebooks

| Notebook | Purpose |
|---|---|
| `preprocessing.ipynb` | Cleans, imputes, scales and splits the IEEE-CIS data; builds the SMOTENC-balanced training pool. Produces `preprocessed.pkl`. |
| `preprocess_patterns.ipynb` | Behavioural feature engineering (time-since-last-transaction, address/product changes, etc.) and five rule-based fraud scenarios, for the dissertation's analysis section. Loads `preprocessed.pkl` — no model training here. |
| `IEEE-CIS-SET1.ipynb` | Main model pipeline on IEEE-CIS with a **held-out real test set and no train/test leakage** (3.5% real fraud rate). This is the headline result. |
| `IEEE-CIS-SET2.ipynb` | Replicates the reference paper's pipeline exactly: SMOTE applied *before* the split, on a 50/50 balanced pool. Kept separate from SET 1 because its test set is SMOTE-contaminated by design — it exists for paper comparison, not as the main result. |
| `paysim.ipynb` | 	Same pipeline as SET 1, run on PaySim. Runs its own dedicated Jaya hyperparameter search on PaySim ("RXT-J + Attention (Jaya)") |



### Run order

```
preprocessing.ipynb
   -> preprocessed.pkl
      -> preprocess_patterns.ipynb   
      -> IEEE-CIS-SET1.ipynb
      -> IEEE-CIS-SET2.ipynb

paysim.ipynb   
```

## Models

1. **Logistic Regression** — baseline.
2. **XGBoost** — baseline.
3. **ResNeXt-GRU** — reference paper's architecture: 4 parallel Dense
   "cardinality" paths, merged and added back (ResNeXt-style residual block),
   fed into a GRU, then a dense classification head.
4. **ResNeXt-GRU + Attention** — same idea with two ResNeXt blocks and a
   `MultiHeadAttention` layer after the GRU. This is the project's own
   architectural contribution on top of the paper.
5. **RXT-J** — the attention model with its hyperparameters (path width, GRU
   units, dropout, learning rate, weight decay, batch size) tuned by the Jaya
   algorithm instead of picked by hand.

All deep models use class weighting (or SMOTE, depending on the pipeline)
plus early stopping and LR reduction on plateau. Decision thresholds are
tuned separately on each test set rather than assumed at 0.5.


## Requirements

```
pandas, numpy, scikit-learn, imbalanced-learn, xgboost, tensorflow,
shap, matplotlib, kagglehub, statsmodels
```

## Reproducing results

Each notebook that trains a deep model saves a `checkpoint_*.pkl` (and
sometimes `.npy` arrays) so that downstream analysis cells (SHAP, threshold
tuning, error analysis, training-time extraction) can be re-run later without
retraining. 