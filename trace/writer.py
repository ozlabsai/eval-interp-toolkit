"""
Serializes SampleTrace results to disk:
  - records.parquet  : one row per sample with scalar stats + per-layer activation norms
  - activations/{sample_id}.pt : full layer activation tensors (dict[int, Tensor])
  - manifest.json    : model identity, run config, file hashes
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import torch

from trace.tracer import HOOK_LAYERS, SampleTrace
from trace.dataset import TraceRecord


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_outputs(
    traces: list[SampleTrace],
    records: list[TraceRecord],
    output_dir: str,
    model_id: str,
    limit: int | None,
    device: str,
) -> None:
    """
    Write all output artefacts to output_dir.
    `records` is parallel to `traces` (same order, same length).
    """
    os.makedirs(output_dir, exist_ok=True)
    activations_dir = os.path.join(output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    _write_parquet(traces, records, output_dir)
    _write_activations(traces, activations_dir)
    _write_manifest(output_dir, model_id, limit, device)


def _write_parquet(
    traces: list[SampleTrace],
    records: list[TraceRecord],
    output_dir: str,
) -> None:
    import pandas as pd

    label_map = {r.sample_id: r for r in records}
    rows = []
    for t in traces:
        rec = label_map.get(t.sample_id)
        row: dict[str, Any] = {
            "sample_id": t.sample_id,
            "label": rec.label if rec else "UNKNOWN",
            "category": rec.category if rec else "unknown",
            "n_windows": t.n_windows,
            "n_tokens": t.n_tokens,
            "mean_entropy": t.mean_entropy,
            "logprob_sum": t.logprob_sum,
            "mean_margin": t.mean_margin,
            "min_margin": t.min_margin,
        }
        # Add per-layer activation L2 norm as a cheap scalar summary
        for layer_idx in HOOK_LAYERS:
            tensor = t.layer_activations.get(layer_idx)
            if tensor is not None:
                row[f"act_norm_l{layer_idx}"] = float(tensor.norm().item())
            else:
                row[f"act_norm_l{layer_idx}"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "records.parquet")
    df.to_parquet(out_path, index=False)


def _write_activations(traces: list[SampleTrace], activations_dir: str) -> None:
    for t in traces:
        if not t.layer_activations:
            continue
        # Save as dict[int, Tensor] — preserves bfloat16, half the size of float32
        out_path = os.path.join(activations_dir, f"{t.sample_id}.pt")
        torch.save(t.layer_activations, out_path)


def _write_manifest(output_dir: str, model_id: str, limit: int | None, device: str) -> None:
    from trace.windower import WINDOW_SIZE, STRIDE

    parquet_path = os.path.join(output_dir, "records.parquet")
    manifest: dict[str, Any] = {
        "model_id": model_id,
        "hook_layers": HOOK_LAYERS,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "limit": limit,
        "device": device,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "records_sha256": _sha256_file(parquet_path) if os.path.exists(parquet_path) else None,
    }
    out_path = os.path.join(output_dir, "manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
