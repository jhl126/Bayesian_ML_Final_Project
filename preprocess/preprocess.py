# %%
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

RAW_DIR = Path("../data/raw")
PREPROCESSED_DIR = Path("../data/preprocessed")

SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

WINSORIZE_COLS = [
    "bank_branch_count_8w",
    "prev_address_months_count",
    "zip_count_4w",
    "days_since_request",
    "session_length_in_minutes",
    "current_address_months_count",
    "velocity_6h",
    "intended_balcon_amount",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "velocity_24h",
]

YEO_JOHNSON_COLS = [
    "bank_branch_count_8w",
    "days_since_request",
    "zip_count_4w",
    "session_length_in_minutes",
    "intended_balcon_amount",
    "bank_months_count",
    "velocity_6h",
]

CAT_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

AGE_BINS = [10, 25, 35, 45, 60, 200]
AGE_LABELS = [0, 1, 2, 3, 4]

CREDIT_LIMIT_ORDER = [190, 200, 500, 1500, 2100]

RANDOM_STATE = 42

# %%
# Load raw data
df = pd.read_csv(RAW_DIR / "Base.csv")
print(f"Shape: {df.shape}")
df.head(3).T

# %%
# Step 1 — Drop zero-variance feature
# device_fraud_count has std=0 across all rows; carries no signal
df = df.drop(columns=["device_fraud_count"])
print(f"Columns after drop: {df.shape[1]}")

# %%
# Step 2 — Add _is_missing flags for sentinel -1 columns
# Missingness is itself a fraud signal (e.g. no prior address = synthetic identity)
# Flags must be created BEFORE imputation so the signal is preserved
for col in SENTINEL_COLS:
    df[f"{col}_is_missing"] = (df[col] == -1).astype(int)

flag_counts = {col: df[f"{col}_is_missing"].sum() for col in SENTINEL_COLS}
print("Missing counts per sentinel column:")
for col, n in flag_counts.items():
    print(f"  {col}: {n:,} ({n / len(df):.1%})")

# %%
# Step 3a — Bin customer_age into ordinal groups
# Age has a non-linear relationship with fraud; bins also give clean nodes for Bayesian CPTs
# [10,25)→0  [25,35)→1  [35,45)→2  [45,60)→3  [60,90]→4
df["customer_age"] = pd.cut(
    df["customer_age"],
    bins=AGE_BINS,
    labels=AGE_LABELS,
    right=False,
).astype(int)

print("customer_age bin distribution:")
print(df["customer_age"].value_counts().sort_index().to_string())

# %%
# Step 3b — Encode proposed_credit_limit as ordered codes
# Values are discrete (190, 200, 500, 1500, 2100) — not truly continuous
# Ordinal codes preserve relative magnitude without implying a linear numeric relationship
cat = pd.Categorical(
    df["proposed_credit_limit"], categories=CREDIT_LIMIT_ORDER, ordered=True
)
df["proposed_credit_limit"] = cat.codes
print("proposed_credit_limit code distribution:")
print(df["proposed_credit_limit"].value_counts().sort_index().to_string())

# %%
# Step 4 — One-hot encode low-cardinality categorical columns + retain label-coded originals
# Done before split so both splits share the same column structure.
# OHE columns (drop_first=True) are used by logistic regression.
# Label-coded originals (integer 0, 1, 2…) are retained under the original column name
# for Bayesian/PyMC models that need discrete category indices via pm.Categorical.
_ohe = pd.get_dummies(df[CAT_COLS], drop_first=True, dtype=int)
for col in CAT_COLS:
    df[col] = pd.Categorical(df[col]).codes
df = pd.concat([df, _ohe], axis=1)
print(f"Shape after one-hot encoding: {df.shape}")
print(
    "OHE columns:",
    [c for c in df.columns if any(c.startswith(cat + "_") for cat in CAT_COLS)],
)

# %%
# Step 5 — 80/20 stratified train/test split
# Split HERE — all fit-dependent steps below must only see training data
X = df.drop(columns=["fraud_bool"])
y = df["fraud_bool"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean():.4f}  Test fraud rate: {y_test.mean():.4f}")

# %%
# Step 6 — Impute sentinel -1s with per-column training median
# Median is robust to the outliers we haven't yet winsorized
medians = {}
for col in SENTINEL_COLS:
    medians[col] = X_train.loc[X_train[col] != -1, col].median()
    X_train[col] = X_train[col].replace(-1, medians[col])
    X_test[col] = X_test[col].replace(-1, medians[col])

print("Imputation medians (from training data):")
for col, val in medians.items():
    print(f"  {col}: {val}")

remaining = {col: (X_train[col] == -1).sum() for col in SENTINEL_COLS}
print(f"\nRemaining -1s in train: {remaining}")

# %%
# Step 7 — Winsorize extreme outliers at 1st/99th percentile (fit on train)
# Critical for logistic regression and autoencoders — outliers distort scaling
# and dominate gradient signals
bounds = {}
for col in WINSORIZE_COLS:
    lo = X_train[col].quantile(0.01)
    hi = X_train[col].quantile(0.99)
    bounds[col] = (lo, hi)
    X_train[col] = X_train[col].clip(lo, hi)
    X_test[col] = X_test[col].clip(lo, hi)

print("Winsorize bounds (train 1st/99th pct):")
for col, (lo, hi) in bounds.items():
    print(f"  {col}: [{lo:.2f}, {hi:.2f}]")

# %%
# Step 8 — Yeo-Johnson power transform for skewed features (fit on train)
# Yeo-Johnson chosen over log1p because it handles zero and negative values natively
transformers = {}
for col in YEO_JOHNSON_COLS:
    pt = PowerTransformer(method="yeo-johnson")
    X_train[col] = pt.fit_transform(X_train[[col]])
    X_test[col] = pt.transform(X_test[[col]])
    transformers[col] = pt

print("Skew after Yeo-Johnson (train):")
print(X_train[YEO_JOHNSON_COLS].skew().round(3).to_string())

# %%
# Step 9 — StandardScaler (fit on train), excluding label-coded categorical columns
# Required for logistic regression convergence, autoencoder gradient stability,
# and well-conditioned MCMC posteriors in PyMC.
# CAT_COLS are excluded: they are discrete integer indices (0, 1, 2…), not continuous
# measurements — scaling them would make them unusable as PyMC category indices.
_scale_cols = [c for c in X_train.columns if c not in CAT_COLS]
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train[_scale_cols]), columns=_scale_cols, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test[_scale_cols]), columns=_scale_cols, index=X_test.index
)
X_train = pd.concat([X_train_scaled, X_train[CAT_COLS]], axis=1)
X_test  = pd.concat([X_test_scaled,  X_test[CAT_COLS]],  axis=1)

print("Train feature stats after scaling (first 5 cols):")
print(X_train.iloc[:, :5].agg(["mean", "std"]).round(3).to_string())

# %%
# Write outputs to parquet
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
X_train.assign(fraud_bool=y_train).to_parquet(
    PREPROCESSED_DIR / "train.parquet", index=False
)
X_test.assign(fraud_bool=y_test).to_parquet(
    PREPROCESSED_DIR / "test.parquet", index=False
)

print(f"Written to {PREPROCESSED_DIR}/")
print(f"  train.parquet — {X_train.shape[0]:,} rows x {X_train.shape[1] + 1} cols")
print(f"  test.parquet  — {X_test.shape[0]:,} rows x {X_test.shape[1] + 1} cols")

# %%
# Write features.json — machine-readable schema mapping group names and per-model feature sets.
# Downstream models load this to select their column subset without hard-coding column names.
#
#   logistic_regression → numeric + OHE columns  (scaled, no bare categoricals)
#   bayesian            → numeric + label-coded original categorical columns
_ohe_cols     = [c for c in X_train.columns if any(c.startswith(cat + "_") for cat in CAT_COLS)]
_numeric_cols = [c for c in X_train.columns if c not in _ohe_cols and c not in CAT_COLS]
features = {
    "numeric":     _numeric_cols,
    "ohe":         _ohe_cols,
    "categorical": list(CAT_COLS),
    "feature_sets": {
        "logistic_regression": _numeric_cols + _ohe_cols,
        "bayesian":            _numeric_cols + list(CAT_COLS),
    },
}
with open(PREPROCESSED_DIR / "features.json", "w") as f:
    json.dump(features, f, indent=2)

print(f"  features.json — {len(_numeric_cols)} numeric, {len(_ohe_cols)} OHE, {len(CAT_COLS)} original categorical")

# %%
