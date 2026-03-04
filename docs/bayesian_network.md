# Part 2 — Bayesian Network Construction

## Overview

Builds two Bayesian Network models for fraud detection using discretized, pre-transform data from the BAF dataset. Compares a Naive Bayes-like structure against a semi-constrained structure learned via HillClimbSearch.

**Outputs:** `data/bayesian_outputs/` — predictions, model files, discretized data, visualizations

---

## Pipeline

```
raw data
  → [1] Pre-transform preprocessing (mirrors Dave's steps 1-6, no scaling)
  → [2] SHAP-based feature selection (top 15 from balanced LogReg)
  → [3] Optimal bin count selection via BIC scoring
  → [4] Quantile-based discretization (fit on train, applied to both)
  → [5A] Model A: Naive Bayes-like structure (all features → fraud_bool)
  → [5B] Model B: Semi-constrained HillClimbSearch (fixed + learned edges)
  → [6] Graph visualization
  → [7-8] Prediction and evaluation on test set
  → [9] Export for Part 3
```

---

## Key Design Decisions

### Why pre-transform discretization?
The BN uses data *before* winsorization, Yeo-Johnson, and StandardScaler. Discretizing raw
(post-imputation) values preserves interpretable bin boundaries — e.g. "income above $50k"
rather than "income above 0.7 standard deviations in transformed space." This is critical for
a model whose value proposition is transparency.

### Why SHAP for feature selection?
Using the SHAP rankings from the balanced logistic regression ensures the BN focuses on
features with demonstrated predictive signal. This also aligns the feature set with the
logistic regression baseline (Dave reruns LR on the same 15 features) for a fair comparison.

### Why two structures?
- **Model A (Naive):** All features are conditionally independent given fraud_bool.
  Simple, fast, fully interpretable. Represents the "transparent baseline" for a regulated
  financial environment.
- **Model B (Semi-constrained):** HillClimbSearch discovers inter-feature edges while
  keeping all feature→fraud edges fixed. Captures potential dependencies (e.g. device signals
  correlated with session behavior) that Model A misses. Comparison reveals whether that
  complexity is justified.

### Bin count selection
BIC score is computed for the naive structure across 2-7 bins. The best BIC is selected
unless the improvement is marginal (<1% over the previous), in which case fewer bins are
preferred for parsimony. This balances model fit against CPD table size.

### Discretization approach
Quantile-based (equal-frequency) binning is used because the raw features have heavy skew
and outliers. Equal-width bins would concentrate most observations in a single bin. Bin edges
are fit on training data only and applied to the test set.

### Structure learning subsampling
HillClimbSearch runs on a 30k-row subsample for tractability. CPDs are then fit on the full
800k training set, so probability estimates reflect all available data.

---

## Output Schema

```
data/bayesian_outputs/
  predictions.parquet        — y_true, y_proba (both models), y_pred at 0.5 threshold
  evaluation_summary.json    — ROC AUC, PR AUC, config, learned edges
  model_naive.bif            — Model A (pgmpy BIF format)
  model_semi.bif             — Model B (pgmpy BIF format)
  train_discretized.parquet  — discretized training data (15 features + fraud_bool)
  test_discretized.parquet   — discretized test data
  selected_features.json     — top 15 feature names (for Dave's LR refit)
  bin_edges.json             — quantile bin edges per continuous feature
  bin_selection_bic.png      — BIC vs bin count plot
  graph_naive.png            — Model A DAG
  graph_semi_constrained.png — Model B DAG
```

---

## Dependencies

Requires everything in the project `requirements.txt`. Key packages:
- `pgmpy` — structure learning, CPD estimation, inference
- `shap` — feature selection
- `scikit-learn` — logistic regression for SHAP, evaluation metrics
- `networkx` / `matplotlib` — graph visualization

## Usage

```bash
cd bayesian/
python bayesian_network.py
```

Expects `data/raw/Base.csv` to exist. Outputs to `data/bayesian_outputs/`.
