"""
Probe direction analysis: extract the EVAL-vs-REAL probe weight vector and
find which tokens most activate it.

Steps:
  1. Fit logistic regression on ALL mean-pooled activations (layer 12, full data)
     to extract the probe weight vector w ∈ ℝ^4096.
  2. For N example samples (EVAL + REAL), re-run a single forward pass on the
     first WINDOW_SIZE tokens and project each token's hidden state onto w.
  3. Report the top-k tokens by projection score, showing what drives the probe.

Usage:
  python -m analysis.probe_direction logs/trace_Mistral-7B-Instruct-v0.3 \\
      --layer 12 --n-examples 6 --top-k 20
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trace.tracer import HOOK_LAYERS, DEFAULT_MODEL_ID, load_model
from trace.windower import _messages_to_text, WINDOW_SIZE
from trace.dataset import load_trace_dataset
from constants import MAIN_DATA_FILE


# ── 1. Fit probe on mean-pooled activations ────────────────────────────────

def fit_probe(trace_dir: str, layer_idx: int) -> tuple[np.ndarray, StandardScaler]:
    """
    Fit logistic regression (EVAL=1, REAL=0) on mean-pooled activations.
    Returns (weight_vector [hidden_dim], fitted scaler).
    Only uses EVAL and REAL samples (excludes HARD_NEG).
    """
    parquet_path = os.path.join(trace_dir, "records.parquet")
    df = pd.read_parquet(parquet_path)
    subset = df[df["label"].isin(["EVAL", "REAL"])].copy()

    acts_dir = os.path.join(trace_dir, "activations")
    vecs, labels = [], []
    for _, row in subset.iterrows():
        pt_path = os.path.join(acts_dir, f"{row['sample_id']}.pt")
        if not os.path.exists(pt_path):
            continue
        layer_acts = torch.load(pt_path, map_location="cpu", weights_only=True)
        if layer_idx not in layer_acts:
            continue
        vecs.append(layer_acts[layer_idx].float().numpy())
        labels.append(1 if row["label"] == "EVAL" else 0)

    X = np.stack(vecs)
    y = np.array(labels)
    print(f"Fitting probe on {len(y)} samples ({y.sum()} EVAL, {(1-y).sum()} REAL) at layer {layer_idx}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_scaled, y)
    # Weight vector in original (unscaled) space: w_orig = w_scaled / scale
    w = clf.coef_[0] / scaler.scale_
    print(f"Probe weight vector norm: {np.linalg.norm(w):.3f}")
    return w, scaler


# ── 2. Token-level projection ──────────────────────────────────────────────

def token_projections(
    messages: list[dict],
    tokenizer,
    model,
    w: np.ndarray,
    layer_idx: int,
    device: str,
    max_tokens: int = WINDOW_SIZE,
) -> tuple[list[str], np.ndarray]:
    """
    Run one forward pass on up to `max_tokens` tokens of the transcript.
    Returns (token_strings, projection_scores) where projection_scores[i] = hidden[i] @ w.
    """
    text = _messages_to_text(messages)
    token_ids = tokenizer.encode(text)[:max_tokens]
    token_strs = [tokenizer.decode([t]) for t in token_ids]

    w_tensor = torch.tensor(w, dtype=torch.float32, device=device)

    captured: dict[int, torch.Tensor] = {}

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dim() == 3:
            hidden = hidden[0]  # [seq_len, hidden_dim]
        captured[layer_idx] = hidden.detach()  # keep on device, full sequence

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(input_ids=input_ids)
    handle.remove()

    # hidden: [seq_len, hidden_dim] → project onto w
    hidden = captured[layer_idx].float()  # [seq_len, hidden_dim]
    scores = (hidden @ w_tensor.unsqueeze(-1)).squeeze(-1).cpu().numpy()  # [seq_len]
    return token_strs, scores


# ── 3. Report ──────────────────────────────────────────────────────────────

def report_top_tokens(
    sample_id: str,
    label: str,
    token_strs: list[str],
    scores: np.ndarray,
    top_k: int,
) -> None:
    print(f"\n  sample_id : {sample_id}  [{label}]")
    print(f"  n_tokens  : {len(token_strs)}  |  mean_score: {scores.mean():.4f}")
    print(f"  {'Rank':<5} {'Score':>7}  Token")
    print(f"  {'─'*5} {'─'*7}  {'─'*30}")
    top_idx = np.argsort(scores)[::-1][:top_k]
    for rank, idx in enumerate(top_idx, 1):
        tok = repr(token_strs[idx])
        print(f"  {rank:<5} {scores[idx]:>7.4f}  {tok}")


def run(trace_dir: str, layer_idx: int, n_examples: int, top_k: int, model_id: str) -> None:
    # Step 1: fit probe
    w, _ = fit_probe(trace_dir, layer_idx)

    # Step 2: load dataset and pick examples
    records = load_trace_dataset(MAIN_DATA_FILE)
    eval_recs  = [r for r in records if r.label == "EVAL"][:n_examples // 2]
    real_recs  = [r for r in records if r.label == "REAL"][:n_examples - len(eval_recs)]
    examples   = eval_recs + real_recs

    print(f"\nLoading model {model_id} for token attribution ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(model_id=model_id, device=device)

    print(f"\n{'═' * 60}")
    print(f"TOKEN ATTRIBUTION  (layer {layer_idx}, top {top_k} tokens per sample)")
    print(f"{'═' * 60}")

    from collections import defaultdict
    token_scores: dict[str, list[float]] = defaultdict(list)

    for rec in examples:
        token_strs, scores = token_projections(
            rec.messages, tokenizer, model, w, layer_idx, device
        )
        report_top_tokens(rec.sample_id, rec.label, token_strs, scores, top_k)
        for tok, score in zip(token_strs, scores):
            token_scores[tok].append(float(score))

    # Step 3: global top tokens — aggregate over all examples
    print(f"\n{'═' * 60}")
    print(f"GLOBAL TOP TOKENS  (aggregated, mean score across all examples where token appears)")
    print(f"{'═' * 60}")

    # Mean score per unique token string, min 2 appearances
    agg = {tok: np.mean(v) for tok, v in token_scores.items() if len(v) >= 2}
    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Token':<30} {'Mean score':>10}  {'Count':>6}")
    print(f"  {'─'*30} {'─'*10}  {'─'*6}")
    for tok, score in ranked[:top_k]:
        count = len(token_scores[tok])
        print(f"  {repr(tok):<30} {score:>10.4f}  {count:>6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--layer", type=int, default=12, choices=HOOK_LAYERS)
    parser.add_argument("--n-examples", type=int, default=6,
                        help="Number of samples to run attribution on (split EVAL/REAL)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top tokens to display per sample")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID,
                        help="HuggingFace model ID (must match trace_dir)")
    args = parser.parse_args()
    run(args.trace_dir, args.layer, args.n_examples, args.top_k, args.model)
