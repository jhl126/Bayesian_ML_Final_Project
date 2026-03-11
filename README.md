# Bayesian Inference for Account Takeover Detection

A Bayesian Network approach to fraud detection using the Bank Account Fraud (BAF) dataset (NeurIPS 2022). This project builds and evaluates two discrete Bayesian Network structures for predicting fraudulent account applications, benchmarked against a logistic regression baseline.

**Course:** ADSP 32014 — Bayesian Machine Learning with GenAI Applications  
**Team:** Dave, Gavin, Josh

---

## Motivation

Fraud detection in banking typically relies on simple, interpretable models that prioritize transparency for auditors and compliance teams. While this interpretability is essential in regulated financial environments, basic models often struggle to capture the complex, conditional relationships between fraud signals, limiting their explanatory depth. Bayesian Networks offer a framework that preserves interpretability through explicit conditional probability estimates, while also uncovering the why behind fraud patterns by modeling how variables interact and influence one another through a directed acyclic graph.

This project investigates whether Bayesian Networks can provide meaningful fraud detection capabilities while maintaining the interpretability that logistic regression and other traditional models offer, and examines how structural assumptions (conditional independence vs. learned dependencies) affect both prediction and inference.

---

## Dataset

The [Bank Account Fraud Dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) contains 1,000,000 anonymized bank account application records with 31 features and a binary fraud label. The fraud rate is approximately 1.1%, reflecting realistic class imbalance.

Features include demographics (age, income, employment status), behavioral signals (session length, device OS, email similarity), credit indicators (credit risk score, proposed credit limit), and account metadata (payment type, housing status).

---

## Project Structure

```
Bayesian_ML_Final_Project/
│
├── data/
│   ├── raw/                          # Base.csv (not tracked — see Setup)
│   ├── preprocessed/                 # Train/test splits from preprocessing
│   │   ├── train.parquet
│   │   ├── test.parquet
│   │   └── features.json
│   └── bayesian_outputs/             # Bayesian Network outputs
│       ├── train_discretized.parquet
│       ├── test_discretized.parquet
│       ├── predictions.parquet
│       ├── predictions_partial.parquet
│       ├── evaluation_summary.json
│       ├── selected_features.json
│       ├── bin_edges.json
│       ├── graph_naive.png
│       ├── graph_semi_constrained.png
│       ├── partial_observation_comparison.png
│       └── bin_selection_bic.png
│
├── preprocess/
│   ├── preprocess.py                 # Data cleaning and feature engineering
│   └── preprocess.md                 # Preprocessing documentation
│
├── feature_importance/
│   ├── feature_importance.py         # SHAP analysis (full feature set)
│   └── feature_importance.ipynb
│
├── baseline/
│   └── logistic_regression_baseline.ipynb  # LR on selected 15 features
│
├── bayesian/
│   ├── bayesian_network.py           # BN construction and evaluation
│   └── bayesian_network.md           # BN documentation
│
├── inference/
│   ├── inference.ipynb               # Variable Elimination and scenario analysis
│   └── inference_md.ipynb            # Writeup and findings
│
├── eda/
│   └── eda.ipynb                     # Exploratory data analysis
│
├── docs/
│   └── bayesian_network.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The directory structure above reflects the logical organization. Some files may be located at slightly different paths in the repository.

---

## Pipeline Overview

The project follows a three-part pipeline:

### Part 1 — Preprocessing and Baseline (Dave)

**Preprocessing** transforms the raw dataset into model-ready train/test splits through nine steps: dropping zero-variance features, creating missingness indicator flags, binning age and credit limit, encoding categoricals, stratified 80/20 splitting, median imputation of sentinel values, winsorization, Yeo-Johnson power transforms, and standard scaling. All fit-dependent transformations are applied to training data only to prevent leakage.

**Feature importance** analysis uses SHAP values on a balanced logistic regression to rank all features by predictive contribution. The top 15 features are selected for the Bayesian Network.

**Baseline logistic regression** is trained on the same 15 features (with OHE expansion for categoricals) using `class_weight="balanced"` to handle the 1.1% fraud rate.

### Part 2 — Bayesian Network Construction (Gavin)

The Bayesian Network pipeline operates on **pre-transform data** — after imputation but before winsorization, Yeo-Johnson, and scaling — to preserve interpretable bin boundaries.

Key steps:

- **Feature selection:** Top 15 features chosen by SHAP importance from the balanced logistic regression
- **Bin optimization:** BIC scoring across 2–7 bins on a naive Bayes structure; BIC selected 2 bins, overridden to 3 (low/medium/high) for domain interpretability
- **Discretization:** Quantile-based binning with edges fit on training data only
- **Model A (Naive Bayes-like):** All 15 features → `fraud_bool` with no inter-feature edges; assumes conditional independence
- **Model B (Semi-Constrained):** HillClimbSearch on a 30k subsample with required edges (all features → `fraud_bool`) plus freedom to discover inter-feature dependencies; learned 24 additional edges
- **CPD estimation:** Maximum Likelihood Estimator on the full 800k training set

### Part 3 — Inference and Evaluation (Josh)

Probabilistic inference using Variable Elimination to query fraud probability under specific scenarios:

- **Persona-based queries:** High-risk profiles, low-risk established customers, and partial observation cases
- **Model comparison:** Demonstrates that both Bayesian models produce identical predictions under full observation (same Markov blanket) but diverge meaningfully under partial observation
- **Posterior predictive checks:** Synthetic data generation from the Semi-Constrained Bayesian Network compared against real training distributions
- **Evaluation metrics:** ROC curves and confusion matrices for the Bayesian Network on the full test set

---

## Key Results

### Classification Performance (Full Observation)

| Model | ROC AUC | PR AUC | Recall (Fraud) |
|---|---|---|---|
| Logistic Regression (balanced, 15 features) | 0.8798 | 0.1261 | 0.8073 |
| Naive Bayesian Network (Model A) | 0.5838 | 0.0133 | 0.2013 |
| Semi-Constrained Bayesian Network (Model B) | 0.5838 | 0.0133 | 0.2013 |

Models A and B produce **identical full-observation predictions** because they share the same Markov blanket for `fraud_bool`. When all 15 features are observed, the inter-feature edges in Model B are d-separated from the target by the evidence.

### Partial Observation (Where the Models Diverge)

Under partial observation (5 of 15 features), the models produce meaningfully different fraud estimates:

| Metric | Value |
|---|---|
| Mean absolute difference | 0.133 |
| Max absolute difference | 0.220 |
| Mean P(fraud) — Naive Bayesian Network | 0.266 |
| Mean P(fraud) — Semi-Constrained Bayesian Network | 0.133 |

The Semi-Constrained Bayesian Network leverages inter-feature dependencies (e.g., `housing_status → income`, `credit_risk_score → employment_status`) to produce more conservative estimates when data is incomplete. Model A, lacking these edges, tends to overestimate fraud risk under uncertainty.

### Variable Elimination Scenarios

| Scenario | P(fraud) | vs. Prior (13.48%) |
|---|---|---|
| Baseline prior (no evidence) | 13.48% | — |
| Persona 1: High-risk profile | 7.64% (Semi-Constrained) / 20.59% (Naive) | Lower than prior |
| Persona 2: Low-risk established customer | 32.64% (Semi-Constrained) | Higher than prior |
| Persona 3: Partial observation (3 features) | 15.66% (Semi-Constrained) / 28.78% (Naive) | Slight increase |

Notable finding: "low-risk" profiles with high income, good credit, and paid email showed *elevated* fraud probability, suggesting the model learned patterns consistent with synthetic identity fraud — where attackers construct convincing high-credibility profiles.

---

## Key Findings and Discussion

**The interpretability–performance tradeoff is real.** The Bayesian Network's ROC AUC (0.58) substantially underperforms logistic regression (0.88) on standard classification. This is driven by CPD sparsity — with 15 parent nodes each having 2–7 states, the fraud CPD table contains millions of cells, many with zero fraud observations at a 1.1% base rate. Discretization also discards information that the continuous logistic regression retains.

**Bayesian Networks offer capabilities logistic regression cannot.** Transparent conditional probability tables, explicit dependency structure via the DAG, and principled reasoning under missing data are valuable in regulated environments where model decisions must be explainable to auditors.

**Structural assumptions matter under incomplete information.** The Markov blanket equivalence under full observation is a theoretically grounded result. The practical value of the semi-constrained structure emerges when analysts have partial information — a realistic scenario in fraud investigation where not all signals are available for every application.

**Generative AI was used as an assistive tool** throughout the project to support code development, debugging, and interpretation of model outputs. All modeling decisions, structure choices, and interpretations were made by the team.

---

## Setup

### Requirements

Python 3.11+ recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

The system `graphviz` binary is also required:

```bash
# macOS
brew install graphviz

# Ubuntu/Debian
apt install graphviz
```

### Data

Download `Base.csv` from the [BAF Dataset on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) and place it in `data/raw/`:

```bash
mkdir -p data/raw
# Move downloaded Base.csv into data/raw/
```

### Running the Pipeline

```bash
# 1. Preprocessing
cd preprocess/
python preprocess.py

# 2. Feature importance and baseline
cd ../feature_importance/
python feature_importance.py

# 3. Bayesian Network construction
cd ../bayesian/
python bayesian_network.py

# 4. Baseline LR on selected features
# Run logistic_regression_baseline.ipynb in Jupyter

# 5. Inference and evaluation
# Run inference.ipynb in Jupyter
```

---

## References

- Jesus, S. et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation.* NeurIPS 2022 Datasets and Benchmarks Track.
- Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques.* MIT Press.
- pgmpy documentation: [https://pgmpy.org](https://pgmpy.org)
