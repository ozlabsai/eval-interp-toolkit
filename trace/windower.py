"""
Converts a list of chat messages into overlapping 512-token windows using the
Mistral tokenizer's chat template.

Strategy: sliding window with stride = window_size // 2 (50% overlap).
Pooling across windows happens in tracer.py (mean pool).

The windower is intentionally decoupled from the model — it only needs the
tokenizer, so it can be called without loading Mistral weights.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F

_ALPHANUMERIC = re.compile(r'^[a-zA-Z0-9]{9}$')


def _normalize_tool_id(tool_id: str) -> str:
    """
    Mistral's chat template requires tool call IDs to be exactly 9 alphanumeric chars.
    Transcripts collected from other models (Claude, GPT-4) use UUIDs or arbitrary strings.
    We hash the original ID to a deterministic 9-char alphanumeric string so that
    assistant tool_calls and tool tool_call_id references stay consistent.
    """
    if _ALPHANUMERIC.match(tool_id):
        return tool_id
    return hashlib.sha256(tool_id.encode()).hexdigest()[:9]


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite tool call IDs in-place (on copies) to satisfy Mistral's template constraints."""
    result = []
    for msg in messages:
        msg = dict(msg)
        # assistant messages: tool_calls list
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            new_calls = []
            for call in msg["tool_calls"]:
                call = dict(call)
                if "id" in call:
                    call["id"] = _normalize_tool_id(call["id"])
                new_calls.append(call)
            msg["tool_calls"] = new_calls
        # tool result messages: tool_call_id field
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            msg["tool_call_id"] = _normalize_tool_id(msg["tool_call_id"])
        result.append(msg)
    return result

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
    messages = _sanitize_messages(messages)
    # Render to string first, then tokenize separately.
    # apply_chat_template with tokenize=True can return a tokenizers.Encoding object
    # from the fast tokenizer backend rather than a plain list[int], which breaks
    # torch.tensor(). Splitting into two steps always yields a plain list.
    text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
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
