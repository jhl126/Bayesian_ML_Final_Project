# Preprocessing Pipeline — Fraud Detection Dataset

## Overview

Transforms `data/raw/Base.csv` (1M rows, 32 columns) into clean, scaled, leakage-free
train/test splits ready for Logistic Regression, Bayesian (pgmpy/PyMC), and Autoencoder models.

**Outputs:** `data/preprocessed/train.parquet`, `data/preprocessed/test.parquet`, `data/preprocessed/features.json`

---

## Pipeline Steps

Steps 1–4 are fit-free (no leakage risk) and run before the split.
Steps 6–8 are fit on training data only, then applied to both splits.

```
raw data
  → [1] Drop zero-variance feature
  → [2] Add _is_missing flags
  → [3] Bin customer_age / encode proposed_credit_limit
  → [4] One-hot encode categoricals + retain label-coded originals
        ↓
  [5] 80/20 stratified train/test split
        ↓  ← fit on train only below ↓
  → [6] Impute sentinel -1s (median)
  → [7] Winsorize (1st / 99th percentile)
  → [8] Yeo-Johnson power transform
  → [9] StandardScaler (numeric + OHE only; label-coded categoricals excluded)
        ↓
  → write train.parquet / test.parquet / features.json
```

---

## Step-by-Step Rationale

### [1] Drop `device_fraud_count`
`device_fraud_count` has zero variance across all 1M rows (std=0, min=max=0). It contributes
no signal to any model and is dropped immediately.

### [2] Sentinel -1 → missing flags + imputation
Five columns use `-1` as a sentinel meaning "data not available":

| Column | % missing (approx) |
|---|---|
| `prev_address_months_count` | ~50 % (median = -1) |
| `bank_months_count` | ~25 % (25th pct = -1) |
| `current_address_months_count` | small fraction |
| `session_length_in_minutes` | small fraction |
| `device_distinct_emails_8w` | small fraction |

**Rationale:** Missingness is itself a fraud signal (e.g. no prior address history correlates
with synthetic identities). A binary `{col}_is_missing` flag is created *before* imputation
so that signal is preserved. The -1s are then replaced with the **median of non-sentinel
training values** (robust to the outliers handled in step 7).

The following columns have genuine negative values and are **not** treated as sentinels:
- `credit_risk_score` (min = -170): actual negative credit scores
- `velocity_6h` (min = -170): refunds/reversals produce negative velocity
- `intended_balcon_amount` (min = -15.5): negative amounts are meaningful

### [3] Binning and ordinal encoding

**`customer_age` → ordinal bins**
Age has a non-linear relationship with fraud risk. Binning into five groups removes noise
from exact ages and produces clean discrete nodes for Bayesian CPTs:

| Bin | Range | Code |
|---|---|---|
| Young adult | [10, 25) | 0 |
| Adult | [25, 35) | 1 |
| Middle-aged | [35, 45) | 2 |
| Senior adult | [45, 60) | 3 |
| Senior | [60, 90] | 4 |

**`proposed_credit_limit` → ordered codes**
Values are effectively discrete (190, 200, 500, 1500, 2100) — not continuous. Treated as an
ordered categorical with codes 0–4 to preserve relative magnitude without implying a linear
numeric relationship.

### [4] One-hot encode categoricals + retain label-coded originals
Five low-cardinality string columns are processed in two ways simultaneously:

`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`

1. **OHE columns** (`drop_first=True`) — e.g. `payment_type_AB`, `payment_type_AC` — used by logistic regression and SHAP. `drop_first` avoids multicollinearity.
2. **Label-coded originals** — the original column (e.g. `payment_type`) is retained as an integer code (0, 1, 2…) for Bayesian/PyMC models that need discrete category indices via `pm.Categorical`.

Both representations co-exist in the parquet. Downstream models select the appropriate
columns via `features.json` (see Output Schema).

This is done before the split so the full category vocabulary is captured and both splits
share the same column structure. No target information is used, so there is no leakage.

Already-binary columns (`email_is_free`, `phone_home_valid`, `phone_mobile_valid`,
`has_other_cards`, `foreign_request`, `keep_alive_session`) are left unchanged.

### [5] 80/20 stratified split
`train_test_split(test_size=0.2, random_state=42, stratify=fraud_bool)` preserves the
~1–2% fraud rate in both splits. All subsequent steps are fitted exclusively on training
data to prevent any leakage of test-set statistics.

### [6] Winsorize extreme outliers
Outliers are clipped to the [1st, 99th] percentile bounds computed on training data. This
is critical for logistic regression (coefficients are sensitive to extreme values) and
autoencoders (reconstruction loss is dominated by outliers).

| Column | Issue |
|---|---|
| `bank_branch_count_8w` | Median=9, max=2385 — extreme sparse right tail |
| `prev_address_months_count` | Max=383, 75th pct=12 after sentinel handling |
| `zip_count_4w` | Max=6700 vs 75th pct=1944 |
| `days_since_request` | Median=0, max=78.5 — near-zero for most rows |
| `session_length_in_minutes` | Max=85.9 vs 75th pct=8.9 |
| `current_address_months_count` | Max=428 vs 75th pct=130 |
| `velocity_6h` | Both tails clipped (min=-170, max=16715) |
| `intended_balcon_amount` | Max=113 vs 75th pct=5 |
| `date_of_birth_distinct_emails_4w` | Max=39 vs 75th pct=13 |
| `credit_risk_score` | Both tails clipped (min=-170, max=389) |
| `velocity_24h` | Upper tail: max=9506 |

### [7] Yeo-Johnson power transform
Applied after winsorizing to reduce residual right-skew and bring distributions closer to
normality. **Yeo-Johnson is used over log1p** because it natively handles zero and negative
values (present in `velocity_6h`, `intended_balcon_amount`, `bank_months_count` after
imputation).

Applied to: `bank_branch_count_8w`, `days_since_request`, `zip_count_4w`,
`session_length_in_minutes`, `intended_balcon_amount`, `bank_months_count`, `velocity_6h`

### [9] StandardScaler
Zero-mean, unit-variance scaling applied to all numeric and OHE columns.
- **Logistic regression**: required for gradient convergence and comparable coefficients
- **PyMC/pgmpy**: well-conditioned posteriors improve MCMC sampling
- **Autoencoder**: stable gradients and consistent reconstruction loss scale

The five label-coded categorical columns (`payment_type`, `employment_status`, etc.) are
**excluded** from scaling — they are discrete integer indices, not continuous measurements,
and scaling them would break their use as `pm.Categorical` indices in PyMC.

Scaler is fitted on training data only and applied to both splits.

---

## Class Imbalance (not handled in this script)

`fraud_bool` has a ~1–2% positive rate. Downstream handling depends on the model:

| Model | Recommended approach |
|---|---|
| Logistic Regression | `class_weight='balanced'` in sklearn |
| Bayesian | Informative priors reflecting rare fraud; or apply SMOTE to X_train/y_train |
| Autoencoder | Train on non-fraud only (anomaly detection); fraud = high reconstruction error |

If using SMOTE, apply it **after** calling `preprocess()` and **only to the training set**.

---

## Usage

Run directly from the `preprocess/` directory:

```bash
python preprocess.py
```

This reads `data/raw/Base.csv`, runs all nine steps, and writes
`train.parquet`, `test.parquet`, and `features.json` to `data/preprocessed/`.

---

## Output Schema

```
data/preprocessed/
  train.parquet   # 800 000 rows
  test.parquet    # 200 000 rows
  features.json   # column lists per model family
```

Both parquet files include all preprocessed feature columns **plus** `fraud_bool` as the
last column. Columns span three groups:

| Group | Examples | Scaled? |
|---|---|---|
| `numeric` | `income`, `velocity_6h`, `credit_risk_score` | Yes |
| `ohe` | `payment_type_AB`, `employment_status_CB` | Yes |
| `categorical` | `payment_type`, `employment_status` (integer codes) | No |

`features.json` maps group names and model-specific column lists so downstream code never
hard-codes column names:

```python
import json
features = json.load(open("data/preprocessed/features.json"))

X_lr    = train[features["feature_sets"]["logistic_regression"]]  # numeric + OHE
X_bayes = train[features["feature_sets"]["bayesian"]]             # numeric + categorical
```
