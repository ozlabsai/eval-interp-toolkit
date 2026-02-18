"""
Converts a list of chat messages into overlapping 512-token windows using the
Mistral tokenizer's chat template.

Strategy: sliding window with stride = window_size // 2 (50% overlap).
Pooling across windows happens in tracer.py (mean pool).

The windower is intentionally decoupled from the model — it only needs the
tokenizer, so it can be called without loading Mistral weights.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """
    Serialize a message list to plain text for tokenization.

    We bypass the model's chat template entirely because transcripts collected
    from Claude/GPT contain roles (tool, mid-conversation system) and tool call
    ID formats that Mistral's strict Jinja template rejects. For activation/logprob
    extraction we only need a consistent token sequence, not valid prompt formatting.

    Format: "<|role|>\n{content}\n" per message, with tool_calls JSON-serialized.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        raw_content = msg.get("content") or ""
        # content may be a string or a list of content blocks (e.g. tool results)
        if isinstance(raw_content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw_content
            )
        else:
            content = str(raw_content)
        # Append tool_calls as JSON if present (agentic transcripts)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            content = (content + "\n" + json.dumps(tool_calls)).strip()
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)

WINDOW_SIZE = 512
STRIDE = WINDOW_SIZE // 2  # 256 tokens


@dataclass
class TokenWindow:
    token_ids: list[int]  # length <= WINDOW_SIZE
    window_idx: int
    start_token: int
    end_token: int


def messages_to_windows(
        messages: list[dict[str, Any]],
        tokenizer,
) -> list[TokenWindow]:
    """
    Apply the Mistral chat template to `messages`, tokenize, then slice into
    overlapping WINDOW_SIZE-token windows.

    Returns at least one window even if the transcript is shorter than WINDOW_SIZE.
    """
    # Normalize tool call IDs before templating — transcripts from Claude/GPT use
    # UUIDs; Mistral's template requires exactly 9 alphanumeric chars.
    # Serialize messages to plain text without using the model's chat template.
    # Mistral's template rejects transcripts from other models (tool roles,
    # mid-conversation system messages, non-9-char tool call IDs). Since we only
    # need a consistent token sequence for the forward pass — not valid Mistral
    # prompt formatting — a simple role-prefixed concatenation is sufficient.
    text = _messages_to_text(messages)
    token_ids: list[int] = tokenizer.encode(text)

    if len(token_ids) == 0:
        return []

    windows: list[TokenWindow] = []
    start = 0
    idx = 0
    while start < len(token_ids):
        end = min(start + WINDOW_SIZE, len(token_ids))
        windows.append(TokenWindow(
            token_ids=token_ids[start:end],
            window_idx=idx,
            start_token=start,
            end_token=end,
        ))
        if end == len(token_ids):
            break
        start += STRIDE
        idx += 1

    return windows


def _compute_window_stats(
        logits: torch.Tensor,  # [1, seq_len, vocab]
        token_ids: torch.Tensor,  # [seq_len]
) -> tuple[float, float, float, float]:
    seq_len = int(token_ids.shape[0])
    if seq_len < 2:
        return 0.0, 0.0, 0.0, 0.0

    lp = F.log_softmax(logits[0, :-1, :], dim=-1)  # [seq_len-1, vocab]
    next_ids = token_ids[1:].to(lp.device)  # [seq_len-1]

    # 1) total logprob of realized continuation
    logprob_sum = lp.gather(1, next_ids[:, None]).sum()

    # 2) mean entropy
    p = lp.exp()
    mean_entropy = (-(p * lp).sum(dim=-1)).mean()

    # 3) margins (top1 - top2) in log-space
    top2 = lp.topk(2, dim=-1).values  # [seq_len-1, 2]
    margins = top2[:, 0] - top2[:, 1]

    mean_margin = margins.mean()
    min_margin = margins.min()

    return (
        float(mean_entropy.item()),
        float(logprob_sum.item()),
        float(mean_margin.item()),
        float(min_margin.item()),
    )
