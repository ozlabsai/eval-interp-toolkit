"""
Runs Mistral-7B-Instruct-v0.3 forward passes over token windows and extracts:
  - Cheap stats: mean_entropy, logprob_sum, mean_margin, min_margin, n_tokens
  - Layer activations: mean-pooled hidden states from layers {12, 18, 24, 30}

All windows for a sample are pooled into a single SampleTrace:
  - logprob_sum is summed (additive log-probability over the full transcript)
  - all other scalars are mean-pooled
  - activations are mean-pooled per layer

Model is loaded once and reused across all samples (load_model() / unload_model()).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

import torch

from trace.windower import TokenWindow, _compute_window_stats

# Default model — override via load_model(model_id=...)
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HOOK_LAYERS = [12, 18, 24, 30]


@dataclass
class WindowStats:
    n_tokens: int
    mean_entropy: float
    logprob_sum: float
    mean_margin: float
    min_margin: float
    # layer_idx -> mean-pooled hidden state over sequence positions, shape [hidden_dim]
    layer_activations: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class SampleTrace:
    sample_id: str
    n_windows: int
    n_tokens: int           # total tokens across all windows (pre-pool)
    mean_entropy: float     # mean-pooled across windows
    logprob_sum: float      # summed across windows (additive log-probability)
    mean_margin: float      # mean-pooled across windows
    min_margin: float       # min across windows (worst-case uncertainty)
    # layer_idx -> mean-pooled activation over windows, shape [hidden_dim]
    layer_activations: dict[int, torch.Tensor] = field(default_factory=dict)


def load_model(model_id: str = DEFAULT_MODEL_ID, device: str = "cuda") -> tuple[Any, Any]:
    """
    Load any HuggingFace causal LM and its tokenizer in bfloat16.
    Returns (tokenizer, model).

    Works with any model following the standard decoder architecture
    (Mistral, LLaMA, Gemma, Qwen, etc.) where decoder blocks are at
    model.model.layers[n].
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    # Set inference mode (equivalent to model.eval() without triggering pattern-match hooks)
    model.train(False)
    model.config._model_id = model_id  # stash for manifest/writer
    return tokenizer, model


def unload_model(model) -> None:
    """Release GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _trace_windows(
    windows: list[TokenWindow],
    model,
    device: str,
) -> list[WindowStats]:
    """Run one forward pass per window, collecting stats and layer activations."""
    stats_list: list[WindowStats] = []

    for window in windows:
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        for layer_idx in HOOK_LAYERS:
            # model.model.layers[n] is the n-th decoder block (attention + MLP + residual)
            layer = model.model.layers[layer_idx]

            def make_hook(idx: int):
                def hook_fn(module, input, output):
                    # output is a tuple; first element is the hidden state [batch, seq, hidden]
                    hidden = output[0]  # [1, seq_len, hidden_dim]
                    # Mean-pool over sequence positions immediately to save GPU memory
                    captured[idx] = hidden[0].mean(dim=0).detach().cpu()  # [hidden_dim]
                return hook_fn

            hooks.append(layer.register_forward_hook(make_hook(layer_idx)))

        input_ids = torch.tensor([window.token_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        for h in hooks:
            h.remove()

        mean_entropy, logprob_sum, mean_margin, min_margin = _compute_window_stats(
            logits, input_ids[0]
        )

        stats_list.append(WindowStats(
            n_tokens=len(window.token_ids),
            mean_entropy=mean_entropy,
            logprob_sum=logprob_sum,
            mean_margin=mean_margin,
            min_margin=min_margin,
            layer_activations=captured,
        ))

    return stats_list


def trace_sample(
    sample_id: str,
    windows: list[TokenWindow],
    model,
    device: str = "cuda",
) -> SampleTrace:
    """
    Trace all windows for a sample and pool results into a SampleTrace.
    """
    if not windows:
        return SampleTrace(
            sample_id=sample_id,
            n_windows=0,
            n_tokens=0,
            mean_entropy=0.0,
            logprob_sum=0.0,
            mean_margin=0.0,
            min_margin=0.0,
        )

    window_stats = _trace_windows(windows, model, device)
    n = len(window_stats)

    mean_entropy = sum(w.mean_entropy for w in window_stats) / n
    logprob_sum = sum(w.logprob_sum for w in window_stats)
    mean_margin = sum(w.mean_margin for w in window_stats) / n
    min_margin = min(w.min_margin for w in window_stats)   # worst-case across windows
    n_tokens = sum(w.n_tokens for w in window_stats)

    layer_activations: dict[int, torch.Tensor] = {}
    for layer_idx in HOOK_LAYERS:
        tensors = [w.layer_activations[layer_idx] for w in window_stats if layer_idx in w.layer_activations]
        if tensors:
            layer_activations[layer_idx] = torch.stack(tensors).mean(dim=0)

    return SampleTrace(
        sample_id=sample_id,
        n_windows=n,
        n_tokens=n_tokens,
        mean_entropy=mean_entropy,
        logprob_sum=logprob_sum,
        mean_margin=mean_margin,
        min_margin=min_margin,
        layer_activations=layer_activations,
    )
