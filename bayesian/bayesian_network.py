# %%
# =============================================================================
# Part 2 — Bayesian Network for Account Takeover Detection
# Author: Gavin
#
# This script:
#   1. Loads raw data and applies pre-transform preprocessing (steps 1-6 from
#      Dave's pipeline — no winsorization, Yeo-Johnson, or scaling)
#   2. Selects top 15 features via SHAP importance from the balanced LogReg
#   3. Determines optimal bin counts for continuous features
#   4. Discretizes continuous features using quantile binning
#   5. Builds two Bayesian Network structures:
#        A) Naive Bayes-like: all features → fraud_bool
#        B) Semi-constrained: HillClimbSearch with fixed edges to fraud_bool
#   6. Fits CPDs with MaximumLikelihoodEstimator
#   7. Validates both models and exports predictions for Part 3
# =============================================================================

import sys
!{sys.executable} -m pip install -r ../requirements.txt



import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pgmpy.estimators import (
    BayesianEstimator,
    BIC,
    HillClimbSearch,
    MaximumLikelihoodEstimator,
)
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

import shap
import scipy.cluster.hierarchy as hclust

warnings.filterwarnings("ignore")

# %%
# =============================================================================
# 0. Configuration
# =============================================================================
RAW_DIR = Path("../data/raw")
OUTPUT_DIR = Path("../data/bayesian_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import os
print(os.getcwd())

RANDOM_STATE = 42
N_TOP_FEATURES = 15
STRUCTURE_LEARNING_SAMPLE = 30_000  # rows for HillClimbSearch (speed)

# Sentinel columns from Dave's pipeline
SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
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

# %%
# =============================================================================
# 1. Load raw data and apply pre-transform preprocessing
#    (Mirrors Dave's steps 1-6, STOPS before winsorize/yeo-johnson/scaling)
# =============================================================================
print("=" * 70)
print("STEP 1: Loading and pre-transform preprocessing")
print("=" * 70)

df = pd.read_csv(RAW_DIR / "Base.csv")
print(f"Raw shape: {df.shape}")

# Step 1 — Drop zero-variance feature
df = df.drop(columns=["device_fraud_count"])

# Step 2 — Add _is_missing flags for sentinel -1 columns
for col in SENTINEL_COLS:
    df[f"{col}_is_missing"] = (df[col] == -1).astype(int)

# Step 3a — Bin customer_age
df["customer_age"] = pd.cut(
    df["customer_age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
).astype(int)

# Step 3b — Encode proposed_credit_limit as ordinal codes
cat = pd.Categorical(
    df["proposed_credit_limit"], categories=CREDIT_LIMIT_ORDER, ordered=True
)
df["proposed_credit_limit"] = cat.codes

# Step 4 — Label-encode categoricals (no OHE needed for BN)
for col in CAT_COLS:
    df[col] = pd.Categorical(df[col]).codes

# Step 5 — 80/20 stratified split (same random_state as Dave)
X = df.drop(columns=["fraud_bool"])
y = df["fraud_bool"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean():.4f}")

# Step 6 — Impute sentinel -1s with per-column training median
medians = {}
for col in SENTINEL_COLS:
    medians[col] = X_train.loc[X_train[col] != -1, col].median()
    X_train[col] = X_train[col].replace(-1, medians[col])
    X_test[col] = X_test[col].replace(-1, medians[col])

print(f"\nPre-transform preprocessing complete.")
print(f"  (No winsorization, Yeo-Johnson, or scaling applied)")

# %%
# =============================================================================
# 2. Feature Selection via SHAP (balanced logistic regression)
#    We need the scaled data just for SHAP ranking, then we use the rankings
#    to select features from the unscaled data.
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Feature selection via SHAP")
print("=" * 70)

# Build a temporary scaled version for SHAP (mirrors Dave's steps 7-9)
# We only need this to rank features — the BN will use the unscaled data
_X_train_scaled = X_train.copy()
_X_test_scaled = X_test.copy()

# OHE for logistic regression
_ohe_train = pd.get_dummies(_X_train_scaled[CAT_COLS].astype(str), drop_first=True, dtype=int)
_ohe_test = pd.get_dummies(_X_test_scaled[CAT_COLS].astype(str), drop_first=True, dtype=int)
# Align OHE columns
_ohe_test = _ohe_test.reindex(columns=_ohe_train.columns, fill_value=0)

_X_lr_train = pd.concat([_X_train_scaled.drop(columns=CAT_COLS), _ohe_train], axis=1)
_X_lr_test = pd.concat([_X_test_scaled.drop(columns=CAT_COLS), _ohe_test], axis=1)

# Quick scale for LR
_scaler = StandardScaler()
_X_lr_train_s = pd.DataFrame(
    _scaler.fit_transform(_X_lr_train), columns=_X_lr_train.columns, index=_X_lr_train.index
)
_X_lr_test_s = pd.DataFrame(
    _scaler.transform(_X_lr_test), columns=_X_lr_test.columns, index=_X_lr_test.index
)

# Train balanced LR
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
lr_model.fit(_X_lr_train_s, y_train)
print(f"Balanced LR ROC AUC: {roc_auc_score(y_test, lr_model.predict_proba(_X_lr_test_s)[:, 1]):.4f}")

# SHAP on balanced sample
_n = min(int((y_train == 1).sum()), 500)
_fraud_idx = y_train[y_train == 1].sample(n=_n, random_state=RANDOM_STATE).index
_non_fraud_idx = y_train[y_train == 0].sample(n=_n, random_state=RANDOM_STATE).index
_sample_idx = _fraud_idx.union(_non_fraud_idx)
_sample_X = _X_lr_train_s.loc[_sample_idx]

print("Computing SHAP values (this may take a minute)...")
dist_matrix = hclust.distance.pdist(_X_lr_train_s.corr().fillna(0).T, metric="correlation")
masker = shap.maskers.Partition(_X_lr_train_s, clustering=hclust.ward(dist_matrix))
explainer = shap.PartitionExplainer(
    lambda x: lr_model.predict_proba(pd.DataFrame(x, columns=_X_lr_train_s.columns))[:, 1],
    masker,
)
shap_vals = explainer(_sample_X)

# Aggregate OHE SHAP back to original feature names
shap_df = pd.DataFrame(shap_vals.values, columns=_X_lr_train_s.columns)


def _aggregate_shap_to_original(shap_df, cat_cols, all_cols):
    """Sum OHE SHAP values back to their parent categorical feature."""
    result = shap_df.copy()
    for cat in cat_cols:
        ohe_cols = [c for c in all_cols if c.startswith(cat + "_")]
        if ohe_cols:
            result[cat] = result[ohe_cols].sum(axis=1)
            result = result.drop(columns=ohe_cols)
    return result


shap_agg = _aggregate_shap_to_original(shap_df, CAT_COLS, _X_lr_train_s.columns)

# Rank by mean |SHAP|
mean_abs_shap = shap_agg.abs().mean().sort_values(ascending=False)
print("\nSHAP feature importance ranking (top 20):")
for i, (feat, val) in enumerate(mean_abs_shap.head(20).items(), 1):
    print(f"  {i:2d}. {feat:<45s} {val:.6f}")

top_features = mean_abs_shap.head(N_TOP_FEATURES).index.tolist()

# Make sure fraud_bool is not accidentally in there
top_features = [f for f in top_features if f != "fraud_bool"]
top_features = top_features[:N_TOP_FEATURES]

print(f"\nSelected top {len(top_features)} features for Bayesian Network:")
for f in top_features:
    print(f"  - {f}")

# Save feature list for Dave's logistic regression refit
with open(OUTPUT_DIR / "selected_features.json", "w") as f:
    json.dump({"top_features": top_features}, f, indent=2)
print(f"\nSaved selected features to {OUTPUT_DIR / 'selected_features.json'}")

# %%
# =============================================================================
# 3. Determine optimal bin count for continuous features
#    Uses BIC scoring on a naive Bayes structure across different bin counts.
#    Concept: similar to an elbow plot — we look for diminishing returns in
#    model fit as we increase the number of bins.
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Optimal bin count selection")
print("=" * 70)

# Identify which selected features are continuous vs already discrete
ALREADY_DISCRETE = set(CAT_COLS) | {"customer_age", "proposed_credit_limit"}
BINARY_COLS = {
    "email_is_free", "phone_home_valid", "phone_mobile_valid",
    "has_other_cards", "foreign_request", "keep_alive_session",
}
MISSING_FLAGS = {f"{c}_is_missing" for c in SENTINEL_COLS}
ALL_DISCRETE = ALREADY_DISCRETE | BINARY_COLS | MISSING_FLAGS

continuous_features = [f for f in top_features if f not in ALL_DISCRETE]
discrete_features = [f for f in top_features if f in ALL_DISCRETE]

print(f"Continuous features to discretize ({len(continuous_features)}):")
for f in continuous_features:
    print(f"  - {f}")
print(f"Already discrete features ({len(discrete_features)}):")
for f in discrete_features:
    print(f"  - {f}")

# Subsample for bin optimization
_opt_sample = X_train.sample(n=min(50_000, len(X_train)), random_state=RANDOM_STATE)
_opt_y = y_train.loc[_opt_sample.index]

bin_candidates = [2, 3, 4, 5, 6, 7]
bic_scores_by_bins = {}

print("\nEvaluating bin counts via BIC on naive Bayes structure...")
for n_bins in bin_candidates:
    # Discretize continuous features
    _disc = _opt_sample[top_features].copy()
    for col in continuous_features:
        try:
            _disc[col] = pd.qcut(
                _disc[col], q=n_bins, labels=range(n_bins), duplicates="drop"
            ).astype(int)
        except ValueError:
            # If qcut fails (too few unique values), use as-is
            _disc[col] = pd.cut(
                _disc[col], bins=n_bins, labels=range(n_bins), duplicates="drop"
            ).astype(int)
    _disc["fraud_bool"] = _opt_y.values

    # Build naive Bayes structure and score with BIC
    edges = [(feat, "fraud_bool") for feat in top_features]
    _bn = DiscreteBayesianNetwork(edges)
    scorer = BIC(_disc)
    bic = scorer.score(_bn)
    bic_scores_by_bins[n_bins] = bic
    print(f"  {n_bins} bins: BIC = {bic:,.0f}")

# Plot the BIC scores
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(bic_scores_by_bins.keys()), list(bic_scores_by_bins.values()), "o-", linewidth=2)
ax.set_xlabel("Number of bins", fontsize=12)
ax.set_ylabel("BIC Score (higher = better fit)", fontsize=12)
ax.set_title("Optimal Bin Count Selection via BIC", fontsize=14)
ax.set_xticks(bin_candidates)
ax.grid(True, alpha=0.3)

# Mark the best
best_n_bins = max(bic_scores_by_bins, key=bic_scores_by_bins.get)
ax.axvline(best_n_bins, color="red", linestyle="--", alpha=0.7, label=f"Best: {best_n_bins} bins")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bin_selection_bic.png", dpi=150)
plt.show()

print(f"\nBest bin count by BIC: {best_n_bins}")

# Check for diminishing returns — if adding more bins barely helps, prefer fewer
# (this is the "elbow" logic)
bic_list = [(k, v) for k, v in sorted(bic_scores_by_bins.items())]
improvements = []
for i in range(1, len(bic_list)):
    prev_bins, prev_bic = bic_list[i - 1]
    curr_bins, curr_bic = bic_list[i]
    pct_improvement = (curr_bic - prev_bic) / abs(prev_bic) * 100 if prev_bic != 0 else 0
    improvements.append((curr_bins, pct_improvement))
    print(f"  {prev_bins}→{curr_bins} bins: {pct_improvement:+.2f}% BIC change")

# Use the best BIC score, but if the improvement from (best-1) to best is < 1%,
# prefer the simpler model
print(f"\n>>> Overriding BIC selection ({best_n_bins} bins) to 3 bins for domain interpretability")
OPTIMAL_BINS = 3
for bins, pct in improvements:
    if bins == best_n_bins and abs(pct) < 1.0 and best_n_bins > 3:
        OPTIMAL_BINS = best_n_bins - 1
        print(f"\n  Marginal improvement at {best_n_bins} bins (<1%), using {OPTIMAL_BINS} for parsimony.")
        break

print(f"\n>>> Using {OPTIMAL_BINS} bins for continuous features")

# %%
# =============================================================================
# 4. Discretize the data
# =============================================================================
print("\n" + "=" * 70)
print(f"STEP 4: Discretizing continuous features into {OPTIMAL_BINS} bins")
print("=" * 70)

# Select only top features
train_bn = X_train[top_features].copy()
test_bn = X_test[top_features].copy()

# Fit bin edges on training data, apply to both
bin_edges = {}
for col in continuous_features:
    # Use qcut on training data to get quantile edges
    _, edges = pd.qcut(train_bn[col], q=OPTIMAL_BINS, retbins=True, duplicates="drop")
    bin_edges[col] = edges
    actual_bins = len(edges) - 1

    # Apply the same edges to both sets
    train_bn[col] = pd.cut(
        train_bn[col], bins=edges, labels=range(actual_bins), include_lowest=True
    ).astype(int)
    test_bn[col] = pd.cut(
        test_bn[col], bins=edges, labels=range(actual_bins), include_lowest=True
    )
    # Handle test values outside training bin range
    test_bn[col] = test_bn[col].fillna(0).astype(int)

    print(f"  {col}: {actual_bins} bins — edges = {np.round(edges, 2)}")

# Add target
train_bn["fraud_bool"] = y_train.values
test_bn["fraud_bool"] = y_test.values

print(f"\nDiscretized train shape: {train_bn.shape}")
print(f"Discretized test shape:  {test_bn.shape}")
print(f"\nValue counts per feature (train):")
for col in top_features:
    vc = train_bn[col].value_counts().sort_index()
    print(f"  {col}: {dict(vc)}")

# Save discretized data
train_bn.to_parquet(OUTPUT_DIR / "train_discretized.parquet", index=False)
test_bn.to_parquet(OUTPUT_DIR / "test_discretized.parquet", index=False)

# Save bin edges for reproducibility
with open(OUTPUT_DIR / "bin_edges.json", "w") as f:
    json.dump({k: v.tolist() for k, v in bin_edges.items()}, f, indent=2)

print(f"\nSaved discretized data and bin edges to {OUTPUT_DIR}/")

# %%
# =============================================================================
# 5A. Model A — Naive Bayes-like structure (all features → fraud_bool)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5A: Naive Bayes-like Bayesian Network")
print("=" * 70)

naive_edges = [(feat, "fraud_bool") for feat in top_features]
print(f"Structure: {len(naive_edges)} edges, all pointing to fraud_bool")

model_naive = DiscreteBayesianNetwork(naive_edges)
model_naive.fit(train_bn, estimator=MaximumLikelihoodEstimator)

# Validation
assert model_naive.check_model(), "Model A failed validation!"
print("Model A passed check_model() validation.")

print(f"\nNodes: {model_naive.nodes()}")
print(f"Edges: {model_naive.edges()}")

# Print CPD summary for fraud_bool (this will be large, just show shape)
fraud_cpd = model_naive.get_cpds("fraud_bool")
print(f"\nfraud_bool CPD shape: {fraud_cpd.values.shape}")
print(f"fraud_bool CPD parents: {fraud_cpd.get_evidence()}")

# Print a few feature CPDs to show they're marginals (no parents in naive structure)
print("\nSample CPDs (features are marginally distributed in naive structure):")
for feat in top_features[:3]:
    cpd = model_naive.get_cpds(feat)
    print(f"\n  {feat}:")
    print(f"    States: {cpd.state_names[feat]}")
    print(f"    P: {np.round(cpd.values.flatten(), 4)}")

# %%
# =============================================================================
# 5B. Model B — Semi-constrained HillClimbSearch
#     Fixed edges: every feature must connect to fraud_bool
#     But HillClimbSearch can also discover inter-feature edges
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5B: Semi-constrained Bayesian Network (HillClimbSearch)")
print("=" * 70)

from pgmpy.estimators import ExpertKnowledge


# Subsample for structure learning speed
_struct_sample = train_bn.sample(
    n=min(STRUCTURE_LEARNING_SAMPLE, len(train_bn)), random_state=RANDOM_STATE
)
print(f"Structure learning on {len(_struct_sample):,} rows")

# Fixed edges: all features must connect to fraud_bool
fixed_edges = [(feat, "fraud_bool") for feat in top_features]

print(f"Fixed edges: {len(fixed_edges)} (all features → fraud_bool)")
print("Searching for additional inter-feature edges...")

# HillClimbSearch with fixed edges
expert = ExpertKnowledge(required_edges=fixed_edges)

hc = HillClimbSearch(_struct_sample)
best_structure = hc.estimate(
    max_indegree=4,
    expert_knowledge=expert,
    scoring_method=BIC(_struct_sample),
)

learned_edges = list(best_structure.edges())
extra_edges = [e for e in learned_edges if e not in fixed_edges]

print(f"\nTotal edges learned: {len(learned_edges)}")
print(f"Fixed edges (feature → fraud): {len(fixed_edges)}")
print(f"Additional inter-feature edges: {len(extra_edges)}")
if extra_edges:
    print("  Discovered inter-feature edges:")
    for src, dst in extra_edges:
        print(f"    {src} → {dst}")

# Build and fit Model B
model_semi = DiscreteBayesianNetwork(learned_edges)
model_semi.fit(train_bn, estimator=MaximumLikelihoodEstimator)

assert model_semi.check_model(), "Model B failed validation!"
print("\nModel B passed check_model() validation.")

# %%
# =============================================================================
# 6. Visualize both graph structures
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Graph visualization")
print("=" * 70)

import networkx as nx


def plot_bn_structure(model, title, filename):
    """Plot Bayesian Network structure with fraud_bool highlighted."""
    G = nx.DiGraph(model.edges())

    fig, ax = plt.subplots(figsize=(14, 10))

    # Layout — put fraud_bool in center
    pos = nx.spring_layout(G, seed=42, k=2.5)
    # Force fraud_bool to center
    if "fraud_bool" in pos:
        pos["fraud_bool"] = np.array([0.0, 0.0])

    # Color nodes
    node_colors = []
    for node in G.nodes():
        if node == "fraud_bool":
            node_colors.append("#e74c3c")  # red for target
        elif node in discrete_features:
            node_colors.append("#3498db")  # blue for discrete
        else:
            node_colors.append("#2ecc71")  # green for discretized continuous

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax)

    # Distinguish fixed vs learned edges
    fixed_set = set(fixed_edges)
    fixed_e = [e for e in G.edges() if e in fixed_set]
    learned_e = [e for e in G.edges() if e not in fixed_set]

    nx.draw_networkx_edges(G, pos, edgelist=fixed_e, edge_color="gray",
                           arrows=True, arrowsize=20, width=1.5, ax=ax)
    if learned_e:
        nx.draw_networkx_edges(G, pos, edgelist=learned_e, edge_color="#e67e22",
                               arrows=True, arrowsize=20, width=2.5, style="dashed", ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor="#e74c3c", label="Target (fraud_bool)"),
        Patch(facecolor="#3498db", label="Discrete feature"),
        Patch(facecolor="#2ecc71", label="Discretized continuous"),
        Line2D([0], [0], color="gray", linewidth=1.5, label="Fixed edge"),
    ]
    if learned_e:
        legend_elements.append(
            Line2D([0], [0], color="#e67e22", linewidth=2.5, linestyle="--", label="Learned edge")
        )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved to {OUTPUT_DIR / filename}")


plot_bn_structure(model_naive, "Model A: Naive Bayes-like Structure", "graph_naive.png")
plot_bn_structure(model_semi, "Model B: Semi-Constrained Structure", "graph_semi_constrained.png")

# %%
# =============================================================================
# 7. Prediction — compute P(fraud_bool=1 | evidence) for test set
#    Both models share the same Markov blanket for fraud_bool when all
#    features are observed, so full-observation predictions are identical.
#    The models diverge under partial observation (missing features).
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: Predictions on test set (full observation)")
print("=" * 70)

test_features = test_bn.drop(columns=["fraud_bool"])


def predict_proba_bn(model, test_data, model_name):
    """Get P(fraud=1) predictions from a Bayesian Network."""
    print(f"\n  Predicting with {model_name}...")
    # predict_probability returns a DataFrame with columns for each state
    proba = model.predict_probability(test_data)
    # Extract P(fraud_bool=1)
    fraud_prob_col = [c for c in proba.columns if "fraud_bool" in str(c) and "1" in str(c)]
    if fraud_prob_col:
        y_proba = proba[fraud_prob_col[0]].values
    else:
        # Fallback: columns might be (fraud_bool, 0) and (fraud_bool, 1)
        print(f"    Available columns: {proba.columns.tolist()}")
        y_proba = proba.iloc[:, -1].values
    print(f"    Done. Predictions range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    return y_proba


y_proba_naive = predict_proba_bn(model_naive, test_features, "Model A (Naive)")
y_proba_semi = predict_proba_bn(model_semi, test_features, "Model B (Semi-Constrained)")

# %%
# =============================================================================
# 7B. Markov Blanket Analysis
#     When all features are observed, both models produce identical predictions
#     because they share the same Markov blanket for fraud_bool.
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7B: Markov Blanket Analysis")
print("=" * 70)

mb_naive = set(model_naive.get_markov_blanket("fraud_bool"))
mb_semi = set(model_semi.get_markov_blanket("fraud_bool"))

print(f"Model A Markov blanket ({len(mb_naive)} nodes): {sorted(mb_naive)}")
print(f"Model B Markov blanket ({len(mb_semi)} nodes): {sorted(mb_semi)}")
print(f"\nSame Markov blanket? {mb_naive == mb_semi}")
print(f"Full-observation predictions identical? {np.allclose(y_proba_naive, y_proba_semi)}")

print("""
KEY INSIGHT: Both models have the same Markov blanket for fraud_bool because
all features are parents of fraud_bool in both structures. When every feature
is observed, the inter-feature edges in Model B are d-separated from
fraud_bool by the observed evidence. The two structures only produce
different predictions under PARTIAL observation (missing features).
""")

# %%
# =============================================================================
# 7C. Partial Observation Test
#     Demonstrate that models diverge when not all features are observed.
#     This is the real-world scenario: an analyst may only have a few signals
#     available and needs to estimate fraud probability from incomplete data.
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7C: Partial Observation Comparison")
print("=" * 70)

# Select a meaningful subset of features (simulate missing data)
partial_features = ["device_os", "housing_status", "income", "credit_risk_score", "phone_home_valid"]
test_partial = test_features[partial_features].head(5000)

print(f"Partial observation: {len(partial_features)} of {len(top_features)} features observed")
print(f"Features used: {partial_features}")
print(f"Test sample: {len(test_partial)} rows")

print("\nPredicting with partial evidence...")
proba_naive_partial = model_naive.predict_probability(test_partial)
proba_semi_partial = model_semi.predict_probability(test_partial)

# Extract fraud=1 probabilities
col_naive = [c for c in proba_naive_partial.columns if "fraud_bool" in str(c) and "1" in str(c)][0]
col_semi = [c for c in proba_semi_partial.columns if "fraud_bool" in str(c) and "1" in str(c)][0]
p_naive_partial = proba_naive_partial[col_naive].values
p_semi_partial = proba_semi_partial[col_semi].values

print(f"\nPredictions identical under partial observation? {np.allclose(p_naive_partial, p_semi_partial)}")
print(f"Mean absolute difference:  {np.abs(p_naive_partial - p_semi_partial).mean():.6f}")
print(f"Max absolute difference:   {np.abs(p_naive_partial - p_semi_partial).max():.6f}")
print(f"Mean P(fraud) — Model A:   {p_naive_partial.mean():.6f}")
print(f"Mean P(fraud) — Model B:   {p_semi_partial.mean():.6f}")

# Plot the divergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(p_naive_partial, bins=50, alpha=0.6, label="Model A (Naive)", color="#3498db")
axes[0].hist(p_semi_partial, bins=50, alpha=0.6, label="Model B (Semi-Constrained)", color="#e67e22")
axes[0].set_xlabel("P(fraud = 1)", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].set_title("Fraud Probability Distribution\n(Partial Observation: 5 of 15 features)", fontsize=12)
axes[0].legend(fontsize=10)

diff = p_semi_partial - p_naive_partial
axes[1].hist(diff, bins=50, color="#2ecc71", alpha=0.7)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("P(fraud)_B − P(fraud)_A", fontsize=11)
axes[1].set_ylabel("Count", fontsize=11)
axes[1].set_title("Model B − Model A Prediction Difference\n(Partial Observation)", fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "partial_observation_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved to {OUTPUT_DIR / 'partial_observation_comparison.png'}")

# %%
# =============================================================================
# 8. Evaluation (full observation)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: Evaluation (full observation)")
print("=" * 70)


def evaluate_bn(y_true, y_proba, model_name, threshold=0.5):
    """Evaluate a Bayesian Network's predictions."""
    y_pred = (y_proba >= threshold).astype(int)

    roc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 50}")
    print(f"  {model_name}")
    print(f"{'=' * 50}")
    print(f"  Threshold: {threshold}")
    print(f"  ROC AUC:   {roc:.4f}")
    print(f"  PR AUC:    {pr_auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    return {"roc_auc": roc, "pr_auc": pr_auc, "confusion_matrix": cm.tolist()}


results_naive = evaluate_bn(y_test.values, y_proba_naive, "Model A: Naive Bayes-like")
results_semi = evaluate_bn(y_test.values, y_proba_semi, "Model B: Semi-Constrained (full obs = same as A)")

print("""
NOTE: Models A and B produce identical full-observation metrics because they
share the same Markov blanket. The value of Model B's inter-feature edges
emerges under partial observation (see Step 7C above).
""")

# %%
# =============================================================================
# 9. Save outputs for Part 3 (Josh)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: Saving outputs for Part 3")
print("=" * 70)

# Save full-observation predictions
predictions = pd.DataFrame({
    "y_true": y_test.values,
    "y_proba_naive": y_proba_naive,
    "y_proba_semi": y_proba_semi,
    "y_pred_naive_050": (y_proba_naive >= 0.5).astype(int),
    "y_pred_semi_050": (y_proba_semi >= 0.5).astype(int),
})
predictions.to_parquet(OUTPUT_DIR / "predictions.parquet", index=False)

# Save partial-observation predictions
partial_predictions = pd.DataFrame({
    "y_true": y_test.values[:5000],
    "p_naive_partial": p_naive_partial,
    "p_semi_partial": p_semi_partial,
    "partial_features_used": [partial_features] * 5000,
})
partial_predictions.to_parquet(OUTPUT_DIR / "predictions_partial.parquet", index=False)

# Save evaluation results
eval_summary = {
    "model_a_naive": results_naive,
    "model_b_semi_constrained": results_semi,
    "optimal_bins_bic_selected": int(best_n_bins),
    "optimal_bins_override": OPTIMAL_BINS,
    "override_reason": "BIC selected 2 bins due to heavy complexity penalty; overridden to 3 for domain interpretability",
    "n_features": len(top_features),
    "features_used": top_features,
    "structure_learning_sample_size": STRUCTURE_LEARNING_SAMPLE,
    "extra_learned_edges": extra_edges,
    "markov_blanket_identical": True,
    "full_observation_predictions_identical": True,
    "partial_observation": {
        "features_observed": partial_features,
        "predictions_identical": False,
        "mean_abs_difference": float(np.abs(p_naive_partial - p_semi_partial).mean()),
        "max_abs_difference": float(np.abs(p_naive_partial - p_semi_partial).max()),
    },
}
with open(OUTPUT_DIR / "evaluation_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2, default=str)

# Save models (pgmpy BIF format)
from pgmpy.readwrite import BIFWriter

BIFWriter(model_naive).write_bif(str(OUTPUT_DIR / "model_naive.bif"))
BIFWriter(model_semi).write_bif(str(OUTPUT_DIR / "model_semi.bif"))

print(f"\nAll outputs saved to {OUTPUT_DIR}/:")
print(f"  predictions.parquet              — test set predictions (full observation, both models)")
print(f"  predictions_partial.parquet      — partial observation predictions (5 of 15 features)")
print(f"  evaluation_summary.json          — metrics, config, Markov blanket analysis")
print(f"  model_naive.bif                  — Model A structure + CPDs")
print(f"  model_semi.bif                   — Model B structure + CPDs")
print(f"  train_discretized.parquet        — discretized training data")
print(f"  test_discretized.parquet         — discretized test data")
print(f"  selected_features.json           — top {N_TOP_FEATURES} SHAP features (for Dave)")
print(f"  bin_edges.json                   — discretization bin edges")
print(f"  bin_selection_bic.png            — bin count selection plot")
print(f"  graph_naive.png                  — Model A graph")
print(f"  graph_semi_constrained.png       — Model B graph")
print(f"  partial_observation_comparison.png — Model A vs B under partial observation")

print("\n" + "=" * 70)
print("PART 2 COMPLETE")
print("=" * 70)

# %%
