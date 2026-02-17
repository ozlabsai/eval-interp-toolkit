"""
Converts a list of chat messages into overlapping 512-token windows using the
Mistral tokenizer's chat template.

Strategy: sliding window with stride = window_size // 2 (50% overlap).
Pooling across windows happens in tracer.py (mean pool).

The windower is intentionally decoupled from the model — it only needs the
tokenizer, so it can be called without loading Mistral weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F

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
    # apply_chat_template handles role formatting and special tokens
    token_ids: list[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

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
