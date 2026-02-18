"""
Logistic regression probes on cheap scalar features from records.parquet.

Comparisons:
  - EVAL vs REAL     (core question)
  - EVAL vs HARD_NEG (harder: eval vs agentic real tasks)

For each comparison, reports:
  - Per-feature AUC (univariate, cross-validated)
  - Combined model AUC (all features, cross-validated)

Usage:
  python -m analysis.probe_cheap logs/trace_Mistral-7B-Instruct-v0.3_mini
  python -m analysis.probe_cheap logs/trace_Mistral-7B-Instruct-v0.3
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


FEATURES = [
    "mean_entropy",
    "mean_margin",
    "logprob_per_token",
    "act_norm_l12",
    "act_norm_l18",
    "act_norm_l24",
    "act_norm_l30",
]

COMPARISONS = [
    ("EVAL vs REAL",     ("EVAL", "REAL")),
    ("EVAL vs HARD_NEG", ("EVAL", "HARD_NEG")),
]


def probe(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> float:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean())


def run(trace_dir: str) -> None:
    parquet_path = f"{trace_dir}/records.parquet"
    df = pd.read_parquet(parquet_path)

    # Derive logprob_per_token if not present (older runs)
    if "logprob_per_token" not in df.columns:
        df["logprob_per_token"] = df["logprob_sum"] / df["n_tokens"].replace(0, np.nan)

    available_features = [f for f in FEATURES if f in df.columns]
    missing = set(FEATURES) - set(available_features)
    if missing:
        print(f"Note: features not in parquet (skipped): {sorted(missing)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for comparison_name, (pos_label, neg_label) in COMPARISONS:
        subset = df[df["label"].isin([pos_label, neg_label])].copy()
        if len(subset) < 10:
            print(f"\n{comparison_name}: too few samples ({len(subset)}), skipping")
            continue

        y = (subset["label"] == pos_label).astype(int).values
        n_pos, n_neg = y.sum(), (1 - y).sum()

        print(f"\n{'─' * 50}")
        print(f"{comparison_name}  (n={len(subset)}: {n_pos} {pos_label}, {n_neg} {neg_label})")
        print(f"{'─' * 50}")
        print(f"{'Feature':<25} {'AUC':>6}")
        print(f"{'─' * 25} {'─' * 6}")

        for feat in available_features:
            x = subset[feat].fillna(0).values.reshape(-1, 1)
            auc = probe(x, y, cv)
            print(f"{feat:<25} {auc:.3f}")

        # Combined model
        X = subset[available_features].fillna(0).values
        auc_combined = probe(X, y, cv)
        print(f"{'─' * 25} {'─' * 6}")
        print(f"{'COMBINED':<25} {auc_combined:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.probe_cheap <trace_dir>")
        sys.exit(1)
    run(sys.argv[1])
