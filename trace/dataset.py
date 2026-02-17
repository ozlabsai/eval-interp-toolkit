"""
Loads eval_awareness_val.json and assigns one of three labels to each sample:
  - "EVAL"     : transcript is from an evaluation context
  - "REAL"     : transcript is from a real deployment (non-agentic)
  - "HARD_NEG" : transcript is from a real deployment but uses agentic scaffold
                 (tool calls, multi-turn task structure) — structurally eval-like but not an eval.

Hard negatives are derived purely from existing metadata:
  eval_category == False AND scaffold is not None
This reuses the same logic as constants.AGENT_CATEGORIES but scoped to NON_EVAL samples.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class TraceRecord:
    sample_id: str
    messages: list[dict[str, Any]]
    label: str          # "EVAL" | "REAL" | "HARD_NEG"
    category: str       # source dataset name (e.g. "cybench", "lmsys")
    eval_category: bool


def load_trace_dataset(filename: str, limit: int | None = None) -> list[TraceRecord]:
    """Return labeled TraceRecords from the given JSON file."""
    with open(filename) as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    records: list[TraceRecord] = []
    for item in data:
        meta = item["metadata"]
        is_eval = bool(meta.get("eval_category", False))
        has_scaffold = meta.get("scaffold") is not None

        if is_eval:
            label = "EVAL"
        elif has_scaffold:
            label = "HARD_NEG"
        else:
            label = "REAL"

        records.append(TraceRecord(
            sample_id=item["id"],
            messages=item["input"],
            label=label,
            category=meta.get("category", "unknown"),
            eval_category=is_eval,
        ))

    return records
