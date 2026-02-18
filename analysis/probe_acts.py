"""
Linear probes on layer activations from activations/*.pt.

For each layer in {12, 18, 24, 30}:
  - Fits a logistic regression on the mean-pooled hidden state [hidden_dim]
  - Reports cross-validated AUC for EVAL vs REAL and EVAL vs HARD_NEG

Also reports:
  - Length-controlled AUC (residualized against log(n_tokens))
  - Within-category AUC (per eval category vs all REAL, tests for category-identity confound)
  - Cross-category AUC (leave-one-eval-category-out, tests generalization)

Usage:
  python -m analysis.probe_acts logs/trace_Mistral-7B-Instruct-v0.3_mini
  python -m analysis.probe_acts logs/trace_Mistral-7B-Instruct-v0.3
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from trace.tracer import HOOK_LAYERS

COMPARISONS = [
    ("EVAL vs REAL",     ("EVAL", "REAL")),
    ("EVAL vs HARD_NEG", ("EVAL", "HARD_NEG")),
]


def probe(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> float:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1),  # regularize — high-dim features
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean())


def residualize(X: np.ndarray, log_n_tokens: np.ndarray) -> np.ndarray:
    """Remove linear effect of log(n_tokens) from each column of X."""
    ctrl = log_n_tokens.reshape(-1, 1)
    lr = LinearRegression().fit(ctrl, X)
    return X - lr.predict(ctrl)


def probe_within_category(
    df: pd.DataFrame,
    layer_acts: dict[int, dict[str, np.ndarray]],
    layer_idx: int,
    min_samples: int = 10,
) -> None:
    """
    For each EVAL category, probe EVAL (that category only) vs ALL REAL.
    High AUC here means the probe works even when EVAL is a single category —
    but it could still be category-identity (the category's domain) rather than
    eval structure. The cross-category probe is the stronger test.
    """
    real_ids  = df[df["label"] == "REAL"]["sample_id"].tolist()
    acts      = layer_acts[layer_idx]
    real_ids  = [sid for sid in real_ids if sid in acts]

    eval_cats = sorted(df[df["label"] == "EVAL"]["category"].unique())

    print(f"\n  [within-category: each EVAL category vs all REAL, layer {layer_idx}]")
    print(f"  {'Category':<30} {'AUC':>6}  {'n_eval':>7}  {'n_real':>7}")
    print(f"  {'─'*30} {'─'*6}  {'─'*7}  {'─'*7}")

    for cat in eval_cats:
        eval_ids = df[(df["label"] == "EVAL") & (df["category"] == cat)]["sample_id"].tolist()
        eval_ids = [sid for sid in eval_ids if sid in acts]
        if len(eval_ids) + len(real_ids) < min_samples or len(eval_ids) < 2:
            continue

        sample_ids = eval_ids + real_ids
        X = np.stack([acts[sid].reshape(-1) for sid in sample_ids])
        y = np.array([1] * len(eval_ids) + [0] * len(real_ids))

        n_splits = min(5, len(eval_ids))  # can't have more folds than positive samples
        if n_splits < 2:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc = probe(X, y, cv)
        print(f"  {cat:<30} {auc:.3f}  {len(eval_ids):>7}  {len(real_ids):>7}")


def probe_cross_category(
    df: pd.DataFrame,
    layer_acts: dict[int, dict[str, np.ndarray]],
    layer_idx: int,
    min_test_samples: int = 5,
) -> None:
    """
    Leave-one-eval-category-out: train probe on EVAL (all other categories) + REAL,
    test on the held-out EVAL category vs REAL.
    This is the strongest test of generalization — if it transfers, the probe has
    learned something structural about evals, not just category-specific vocabulary.
    """
    acts     = layer_acts[layer_idx]
    real_ids = [sid for sid in df[df["label"] == "REAL"]["sample_id"] if sid in acts]
    eval_df  = df[df["label"] == "EVAL"].copy()
    eval_cats = sorted(eval_df["category"].unique())

    print(f"\n  [cross-category: leave-one-eval-category-out, layer {layer_idx}]")
    print(f"  {'Held-out category':<30} {'AUC':>6}  {'n_test':>7}  {'n_train_eval':>13}")
    print(f"  {'─'*30} {'─'*6}  {'─'*7}  {'─'*13}")

    for held_out in eval_cats:
        test_eval_ids  = [sid for sid in eval_df[eval_df["category"] == held_out]["sample_id"] if sid in acts]
        train_eval_ids = [sid for sid in eval_df[eval_df["category"] != held_out]["sample_id"] if sid in acts]

        if len(test_eval_ids) < min_test_samples or len(train_eval_ids) < 5:
            continue

        # Training set: all other EVAL categories + all REAL
        X_train = np.stack(
            [acts[sid].reshape(-1) for sid in train_eval_ids] +
            [acts[sid].reshape(-1) for sid in real_ids]
        )
        y_train = np.array([1] * len(train_eval_ids) + [0] * len(real_ids))

        # Test set: held-out EVAL + all REAL
        X_test = np.stack(
            [acts[sid].reshape(-1) for sid in test_eval_ids] +
            [acts[sid].reshape(-1) for sid in real_ids]
        )
        y_test = np.array([1] * len(test_eval_ids) + [0] * len(real_ids))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1))
        pipe.fit(X_train, y_train)
        scores = pipe.predict_proba(X_test)[:, 1]

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, scores)
        print(f"  {held_out:<30} {auc:.3f}  {len(test_eval_ids):>7}  {len(train_eval_ids):>13}")


def load_activations(trace_dir: str, sample_ids: list[str]) -> dict[int, dict[str, np.ndarray]]:
    """
    Returns {layer_idx: {sample_id: activation_vector}}.
    Samples with no .pt file are skipped.
    """
    acts_dir = os.path.join(trace_dir, "activations")
    result: dict[int, dict[str, np.ndarray]] = {l: {} for l in HOOK_LAYERS}

    for sample_id in sample_ids:
        pt_path = os.path.join(acts_dir, f"{sample_id}.pt")
        if not os.path.exists(pt_path):
            continue
        layer_acts: dict[int, torch.Tensor] = torch.load(pt_path, map_location="cpu", weights_only=True)
        for layer_idx, tensor in layer_acts.items():
            result[layer_idx][sample_id] = tensor.float().numpy()

    return result


def run(trace_dir: str) -> None:
    parquet_path = os.path.join(trace_dir, "records.parquet")
    df = pd.read_parquet(parquet_path)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Load all activations upfront
    all_sample_ids = df["sample_id"].tolist()
    print(f"Loading activations for {len(all_sample_ids)} samples ...")
    layer_acts = load_activations(trace_dir, all_sample_ids)

    n_loaded = len(next(iter(layer_acts.values())))
    print(f"Loaded activations: {n_loaded} samples across layers {HOOK_LAYERS}")
    # Sanity check shape of first loaded activation
    for layer_idx, acts in layer_acts.items():
        if acts:
            first = next(iter(acts.values()))
            print(f"  layer {layer_idx} activation shape: {first.shape}")
            break

    for comparison_name, (pos_label, neg_label) in COMPARISONS:
        subset = df[df["label"].isin([pos_label, neg_label])].copy()
        if len(subset) < 10:
            print(f"\n{comparison_name}: too few samples ({len(subset)}), skipping")
            continue

        y_map = {sid: int(lbl == pos_label)
                 for sid, lbl in zip(subset["sample_id"], subset["label"])}

        print(f"\n{'─' * 50}")
        print(f"{comparison_name}  (n={len(subset)})")
        print(f"{'─' * 50}")
        print(f"{'Layer':<20} {'AUC':>6}  {'n_samples':>10}")
        print(f"{'─' * 20} {'─' * 6}  {'─' * 10}")

        per_layer_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for layer_idx in HOOK_LAYERS:
            acts = layer_acts[layer_idx]
            # Intersect: samples that have both a label and an activation
            common_ids = [sid for sid in subset["sample_id"] if sid in acts]
            if len(common_ids) < 10:
                print(f"  layer {layer_idx:<14} {'n/a':>6}  {len(common_ids):>10}")
                continue

            vecs = [acts[sid].reshape(-1) for sid in common_ids]
            X = np.stack(vecs)   # [n_samples, hidden_dim]
            y = np.array([y_map[sid] for sid in common_ids])
            per_layer_arrays[layer_idx] = (X, y)

            auc = probe(X, y, cv)
            print(f"  layer {layer_idx:<14} {auc:.3f}  {len(common_ids):>10}")

        # Concatenated probe across all available layers
        if len(per_layer_arrays) > 1:
            # Use only samples present in ALL layers
            common_all = sorted(
                set(subset["sample_id"]).intersection(
                    *[set(layer_acts[l].keys()) for l in per_layer_arrays]
                )
            )
            if len(common_all) >= 10:
                # [n_samples, n_layers * hidden_dim]
                X_cat = np.stack([
                    np.concatenate([layer_acts[l][sid].reshape(-1) for l in sorted(per_layer_arrays)])
                    for sid in common_all
                ])
                y_cat = np.array([y_map[sid] for sid in common_all])
                auc_cat = probe(X_cat, y_cat, cv)
                print(f"{'─' * 20} {'─' * 6}  {'─' * 10}")
                print(f"  {'CONCATENATED':<18} {auc_cat:.3f}  {len(common_all):>10}")

                # Length-controlled: residualize activations against log(n_tokens)
                len_map = dict(zip(subset["sample_id"], subset["n_tokens"].fillna(0)))
                log_len_cat = np.log1p(np.array([len_map[sid] for sid in common_all], dtype=float))
                print(f"\n  [length-controlled: residualized against log(n_tokens)]")
                print(f"  {'Layer':<18} {'AUC':>6}  {'n_samples':>10}")
                print(f"  {'─' * 18} {'─' * 6}  {'─' * 10}")
                for layer_idx, (X_l, y_l) in sorted(per_layer_arrays.items()):
                    common_l = [sid for sid in subset["sample_id"] if sid in layer_acts[layer_idx]]
                    if len(common_l) < 10:
                        continue
                    log_len_l = np.log1p(np.array([len_map[sid] for sid in common_l], dtype=float))
                    X_l_res = residualize(X_l, log_len_l)
                    auc_l_res = probe(X_l_res, y_l, cv)
                    print(f"  layer {layer_idx:<12} {auc_l_res:.3f}  {len(common_l):>10}")
                X_cat_res = residualize(X_cat, log_len_cat)
                auc_cat_res = probe(X_cat_res, y_cat, cv)
                print(f"  {'─' * 18} {'─' * 6}  {'─' * 10}")
                print(f"  {'CONCATENATED':<18} {auc_cat_res:.3f}  {len(common_all):>10}")

    # Category confound analysis — run once, on layer 12 (best single-layer AUC)
    run_category_probes(df, layer_acts, layer_idx=12)


def run_category_probes(
    df: pd.DataFrame,
    layer_acts: dict[int, dict[str, np.ndarray]],
    layer_idx: int = 12,
) -> None:
    """Run within-category and cross-category probes (EVAL vs REAL only)."""
    subset = df[df["label"].isin(["EVAL", "REAL"])]
    if len(subset) < 10:
        print("\nSkipping category probes: too few EVAL+REAL samples")
        return

    print(f"\n{'═' * 60}")
    print(f"CATEGORY CONFOUND ANALYSIS  (layer {layer_idx})")
    print(f"{'═' * 60}")
    probe_within_category(df, layer_acts, layer_idx)
    probe_cross_category(df, layer_acts, layer_idx)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.probe_acts <trace_dir>")
        sys.exit(1)
    run(sys.argv[1])
