# Results

Summary of the results reported in the dissertation and the
project presentation. 

## Headline result

**RXT-J+Attention, the Jaya-tuned model achieves the best accuracy and
AUC-ROC of any model on IEEE-CIS SET1**, the leakage-free, realistic-fraud-rate
benchmark: **91.10% accuracy,
0.928 AUC-ROC**, ahead of the manually-configured Attention model
(90.66% / 0.925) and the base paper's plain ResNeXt-GRU (89.5% / 0.927).

## SET1 vs SET2

- **SET1** - SMOTE applied *after* the train/test split. Test set is 30,000
  real transactions at IEEE-CIS's natural 3.58% fraud rate. No leakage. This
  is the project's primary, realistic benchmark.
- **SET2** - replicates the base paper's own pipeline: SMOTE applied
  *before* the split, on a 50/50 balanced pool (24,000-row test set),
  for comparison against the paper - its test set is
  SMOTE-contaminated by design, so numbers are inflated relative to SET1.

Every model scores higher on SET2 than on SET1 under the *same*
architecture - XGBoost: 92.1% (SET2) vs 90.0% (SET1); ResNeXt-GRU:
90.7% (SET2) vs 89.5% (SET1). This gap proves:
that a meaningful share of the accuracy figures
reported in fraud-detection literature (including the base paper's ~98%)
can be a result from how the test set was built.
RXT-J+Attention was tuned and evaluated only on SET1.

## IEEE-CIS test-set results

Accuracy / AUC-ROC at the default 0.5 threshold:

| Variation | Model | Accuracy | AUC-ROC |
|---|---|---|---|
| SET1 (3.58% real fraud, 30K test) | Logistic Regression | 83.7% | 0.871 |
| | ResNeXt-GRU | 89.5% | 0.927 |
| | ResNeXt-GRU + Attention | 90.66% | 0.925 |
| | XGBoost | 90.0% | 0.918 |
| | **RXT-J (Jaya) + Attention** | **91.10%** | **0.928** |
| SET2 (50% fraud, SMOTE-before-split, 24K test) | Logistic Regression | 81.0% | 0.893 |
| | ResNeXt-GRU | 90.7% | 0.967 |
| | ResNeXt-GRU + Attention | 90.91% | 0.968 |
| | XGBoost | 92.1% | 0.973 |

Precision / Recall / F1 at each model's own best threshold (0.1–0.9;
baselines LR/XGBoost were not threshold-tuned, so SET1, SET2 rows below are
at the default 0.5):

| Variation | Model | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|
| SET1 | Logistic Regression | 0.5 | 0.148 | 0.745 | 0.247 |
| | ResNeXt-GRU | 0.9 | 0.491 | 0.533 | 0.561 |
| | ResNeXt-GRU + Attention | 0.9 | 0.578 | 0.580 | 0.579 |
| | XGBoost | 0.5 | 0.233 | 0.783 | 0.359 |
| | RXT-J (Jaya) | 0.9 | 0.562 | 0.603 | 0.581 |
| SET2 | Logistic Regression | 0.5 | 0.836 | 0.773 | 0.804 |
| | ResNeXt-GRU | 0.5 | 0.904 | 0.910 | 0.907 |
| | ResNeXt-GRU + Attention | 0.5 | 0.898 | 0.923 | 0.910 |
| | XGBoost | 0.5 | 0.930 | 0.912 | 0.921 |

On SET1, XGBoost remains a genuinely difficult benchmark for the deep
models to beat - tree-based splits are naturally suited to the sparse,
high-cardinality, one-hot encoded feature space this pipeline produces
(1,670 columns after encoding).

## PaySim (independent cross-dataset validation)

Same five architectures, same leakage-free split methodology as SET1,
evaluated on PaySim's natural 0.129% fraud rate (30K test). RXT-J's Jaya
search was **re-run independently on PaySim** rather than reusing the
IEEE-CIS hyperparameters, since the two datasets differ in scale and
feature composition that one dataset's best parameters aren't guaranteed to transfer.

Accuracy / AUC-ROC at 0.5 threshold:

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 96.63% | 0.9904 |
| XGBoost | 99.48% | 1.0000 |
| ResNeXt-GRU | 98.99% | 1.0000 |
| ResNeXt-GRU + Attention | 98.96% | 0.9998 |
| **RXT-J (Jaya) + Attention** | **99.30%** | **1.0000** |

Precision / Recall / F1 at best threshold (swept 0.1–0.99):

| Model | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.99 | 61.36% | 58.97% | 71.88% |
| XGBoost | 0.99 | 66.10% | 100.00% | 79.59% |
| ResNeXt-GRU | 0.99 | 92.1% | 82.05% | 90.14% |
| ResNeXt-GRU + Attention | 0.99 | 91.89% | 87.18% | 89.47% |
| **RXT-J (Jaya) + Attention** | 0.99 | **94.59%** | 89.74% | **92.11%** |

**Cross-dataset pattern reversal:** on IEEE-CIS, the manually-configured
Attention model already beats the plain ResNeXt-GRU before any tuning. On
PaySim, that reverses - Attention underperforms plain ResNeXt-GRU by very small value and only
the Jaya-tuned RXT-J+Attention regains the advantage. This suggests
Attention's edge on IEEE-CIS is tied to that dataset's much larger feature
space (1,600+ one-hot columns vs PaySim's less number of fields) - the benefit
of tuning matters more when the feature space is smaller.

## Error analysis (threshold = 0.9)

Confusion-matrix breakdown for Attention and RXT-J+Attention on SET1 and
PaySim, at a common threshold so the error*types are comparable:

| | Attention (SET1) | RXT-J+Attention (SET1) | Attention (PaySim) | RXT-J+Attention (PaySim) |
|---|---|---|---|---|
| True Positives | 606 | 648 | 39 | 39 |
| False Negatives | 468 | 426 | 0 | 0 |
| False Positives | 381 | 558 | 312 | 510 |
| True Negatives | 28,545 | 28,368 | 29,649 | 29,451 |
| Recall | 58.00% | 60.3% | 97.4% | 100% |
| Precision | 57.8% | 56.2% | 55.0% | 54.9% |

At the same threshold, RXT-J catches slightly more fraud than Attention on
SET1 (60.3% vs 58.0% recall), while Attention holds a marginally higher
precision on both datasets. Recall on PaySim is far higher for both models
(≥97%) than on SET1 (~58–60%), reflecting PaySim's more extreme class
imbalance and simpler decision boundary. Lowering the SET1 threshold from
0.9 to 0.5 raises Attention's recall from ~58% to over 81%, but pushes false
alarms from 454 to 2,749 - the precision/recall trade-off, and ultimately a
deployment decision for whoever's paying for missed fraud vs false alarms.

## Explainability (SHAP)

SHAP (`GradientExplainer`, 200-row background / 50-row explanation sample)
was run on all three IEEE-CIS SET1 deep models (ResNeXt-GRU, Attention,
RXT-J).

- **TransactionAmt** is the most influential feature for every
  model - high amounts push toward fraud, low amounts toward legitimate.
  That separation is visibly cleaner for Attention and RXT-J than for the
  plain ResNeXt-GRU.
- **TransactionAmt**, **TransactionDT** (timestamp) and **card3** rank in
  the top features across all three independently-trained models -
  evidence the models converged on a stable, non-spurious signal rather
  than noise.
- One-hot encoded **card6_credit** (credit vs debit flag) ranks in the top
  10 for Attention and RXT-J, generally pushing toward fraud.
- Several of IEEE-CIS's undocumented "V"-prefixed engineered features
  (e.g. V283, V94) rank highly - SHAP can show in which
  direction they matter, but not what they represent, since Vesta doesn't
  publish their definitions.

## Behavioural pattern analysis (rule-based, on raw features)

Separate from the models, `preprocess_patterns.ipynb` surfaces interpretable
fraud signals directly from the raw transaction fields:

- **Card network:** Discover cards have the highest fraud rate (7.73%),
  well above Visa (3.48%), Mastercard (3.43%) and Amex (2.87%).
- **Credit vs debit:** credit cards show a much higher fraud rate (6.68%)
  than debit cards (2.43%).
- **Behavioural signals:** large deviations from a card's historical
  spending, transactions shortly after a long dormant period (>90 days),
  and cards used across an unusually high number of unique addresses are
  all associated with elevated fraud rates - the last one consistent with
  cloned-card use across locations.

These findings also explain, in plain terms, on why SHAP ranks
`TransactionAmt`, `card4` and `card6` so highly for the black-box models.

## Comparison against the base paper

The base paper (Almazroi and Ayub, 2023) reports ~98% accuracy on IEEE-CIS
using RXT-J. Re-implementing both their architecture and their
SMOTE-before-split methodology (SET2) reproduces inflated numbers in the
same direction (e.g. XGBoost 92.1% / AUC 0.973 on SET2 vs 90.0% / 0.918 on
leakage-free SET1). RXT-J+Attention's realistic figure on SET1 (91.1%
accuracy) is lower than the paper's headline number but reflects a model
evaluated under conditions closer to real deployment - no test-set
contamination, and a natural (not rebalanced) fraud rate.

## Limitations

- The Jaya hyperparameter search space and population size were constrained
  by available compute (Population=6, Generations=4) - a broader search was
  out of scope.
- This is a single replication of the base paper's methodology, not
  multiple runs with variance reported.
- Baseline models (Logistic Regression, XGBoost) were evaluated at the
  default 0.5 threshold in Table 4.3/5.1 rather than independently
  threshold-tuned at that stage (threshold tuning is reported separately in
  document Table 4.4/5.2).
