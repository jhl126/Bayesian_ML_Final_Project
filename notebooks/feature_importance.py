# %%
import json

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as hclust
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

# %%
# 0. Read data
with open("../data/preprocessed/features.json") as f:
    features = json.load(f)

lr_cols = features["feature_sets"]["logistic_regression"]

train = pd.read_parquet("../data/preprocessed/train.parquet")
test  = pd.read_parquet("../data/preprocessed/test.parquet")

# %%
# 1. Setup Data
X = train[lr_cols]
y = train["fraud_bool"]
X_test = test[lr_cols]
y_test = test["fraud_bool"]
sample_X = X.sample(n=1000, random_state=42)
sample_y = y.loc[sample_X.index]
counts = sample_y.value_counts().sort_index()
print(f"SHAP sample class balance — 0: {counts[0]}, 1: {counts[1]} ({counts[1] / len(sample_y):.2%} fraud)")


# %%
# 2. Define Grouping Logic
CAT_COLS = features["categorical"]


def get_feature_groups(columns, cat_cols):
    groups = {}
    for col in columns:
        matched = next((cat for cat in cat_cols if col.startswith(cat + "_")), None)
        key = matched if matched else col
        groups.setdefault(key, []).append(col)
    return groups


feature_groups = get_feature_groups(X.columns, CAT_COLS)


# %%
# 3. Build Masker (once — depends only on X, not on model)
dist_matrix = hclust.distance.pdist(X.corr().fillna(0).T, metric="correlation")
masker = shap.maskers.Partition(X, clustering=hclust.ward(dist_matrix))


# %%
# 4. Helper to Train and Aggregate SHAP
def get_aggregated_shap(weight_option, masker):
    model = LogisticRegression(max_iter=1000, class_weight=weight_option)
    model.fit(X, y)

    explainer = shap.PartitionExplainer(
        lambda x: model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1], masker
    )
    shap_vals = explainer(sample_X)

    # Aggregate OHE columns back to original feature granularity
    shap_df = pd.DataFrame(shap_vals.values, columns=X.columns)
    data_df = sample_X.copy()
    for prefix, cols in feature_groups.items():
        if len(cols) > 1:
            shap_df[prefix] = shap_df[cols].sum(axis=1)
            shap_df = shap_df.drop(columns=cols)
            # Encode active OHE category as an ordinal code for colour axis
            # 0 = baseline (dropped first dummy), 1/2/3… = each active OHE col
            codes = pd.Series(0, index=data_df.index)
            for i, col in enumerate(cols, start=1):
                codes[data_df[col] == 1] = i
            data_df[prefix] = codes.values
            data_df = data_df.drop(columns=cols)

    return shap.Explanation(
        values=shap_df.values,
        base_values=shap_vals.base_values,
        data=data_df.values,
        feature_names=shap_df.columns.tolist(),
    ), model


# %%
# 5. Compare Models
print("Training standard model...")
shap_standard, model_standard = get_aggregated_shap(None, masker)

print("Training balanced model...")
shap_balanced, model_balanced = get_aggregated_shap("balanced", masker)


# %%
# 6. Evaluate Models
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR AUC:   {average_precision_score(y_test, y_proba):.4f}")
    print(f"MCC:      {matthews_corrcoef(y_test, y_pred):.4f}")

    _, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title(name)
    plt.tight_layout()
    plt.show()


evaluate_model(model_standard, X_test, y_test, "Standard LogReg")
evaluate_model(model_balanced, X_test, y_test, "Balanced LogReg")

# %%
# 7. Visualization
shap.plots.beeswarm(shap_standard, max_display=15, show=False)
plt.title("Standard Logistic Regression")
plt.tight_layout()
plt.show()

shap.plots.beeswarm(shap_balanced, max_display=15, show=False)
plt.title("Balanced Logistic Regression")
plt.tight_layout()
plt.show()

# %%
