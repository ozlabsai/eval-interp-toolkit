import argparse
import os
import subprocess
import sys

from constants import MAIN_DATA_FILE, find_most_recent_file

STAGES = ["eval_mcq", "purpose_mcq", "purpose_open", "eval_mcq_cot", "purpose_open_cot", "plot_roc", "plot_calib", "plot_purpose", "plot_cot", "print_table", "trace_local", "probe_cheap", "probe_acts"]

MODELS = [
    ("openai", "gpt-4.1-mini-2025-04-14", None, None),
    ("openai", "gpt-4.1-2025-04-14", None, None),
    ("openai", "gpt-4o-2024-11-20", None, None),
    ("openai", "o4-mini-2025-04-16", None, "medium"),
    ("openrouter/google", "gemini-2.5-pro-preview-03-25", None, None),
    ("openrouter/google", "gemini-2.5-flash-preview", None, None),
    ("anthropic", "claude-3-7-sonnet-20250219", None, None),
    ("anthropic", "claude-3-7-sonnet-20250219", "16384", None),
    ("anthropic", "claude-3-5-sonnet-20241022", None, None),
]


def find_model(model_ix: int, experiment_dir: str) -> tuple[str, str, list[str]]:
    provider, model, reasoning_tokens, reasoning_effort = MODELS[model_ix]
    if reasoning_tokens is not None:
        model_name = f"{model}-{reasoning_tokens}"
        thinking_param = ["--reasoning-tokens", str(reasoning_tokens)]
    elif reasoning_effort is not None:
        model_name = f"{model}-{reasoning_effort}"
        thinking_param = ["--reasoning-effort", str(reasoning_effort)]
    else:
        model_name = model
        thinking_param = []
    return f"{provider}/{model}", f"logs/{experiment_dir}/{model_name}", thinking_param


def eval_or_purpose_mcq(model_ixs: list[int], purpose: bool, openended: bool, limit: int, experiment_dir: str, epochs: int):
    for model_ix in model_ixs:
        full_model, output_log_dir, thinking_param = find_model(model_ix, experiment_dir)
        if purpose:
            if openended:
                pq_file = "pq_ablation/purpose_open.yaml"
            else:
                pq_file = "pq_ablation/purpose_mcq.yaml"
            purpose_flag = ["-Tis_purpose=true"]
        else:
            assert not openended, "open-ended evaluation is only for purpose"
            pq_file = "pq_ablation/eval_mcq.yaml"
            purpose_flag = []

        command = [
            "inspect",
            "eval",
            "pipeline.py",
            "--model",
            full_model,
            f"-Tfilename={MAIN_DATA_FILE}",
            f"-Tpq_filename={pq_file}",
            "--max-connections",
            "100",
            "--no-fail-on-error",
            "--display",
            "plain",
            "--log-dir",
            output_log_dir,
            "--limit",
            str(limit),
            *thinking_param,
            *purpose_flag,
            "--epochs",
            str(epochs),
        ]
        subprocess.run(command, check=True)


def eval_mcq_cot(model_ixs: list[int], input_experiment_dir: str, output_experiment_dir: str, scorer: str):
    for model_ix in model_ixs:
        _, input_log_dir, _ = find_model(model_ix, input_experiment_dir)
        _, output_log_dir, _ = find_model(model_ix, output_experiment_dir)
        unique_file = find_most_recent_file(input_log_dir)
        if unique_file is None:
            raise ValueError(f"No unique .eval file found in {input_log_dir}")

        if scorer == "reasoning_classifier":
            pq_args = ["-Tpqs=direct_a,h_purpose_analysis_a"]
        else:
            pq_args = []

        command = [
            "inspect",
            "eval",
            "cot_classifiers/cot_classifiers.py",
            f"-Tfilename={unique_file}",
            "--max-connections",
            "100",
            "--model",
            "openai/gpt-4.1",
            f"-Tscorers={scorer}",
            *pq_args,
            "--display",
            "plain",
            "--log-dir",
            output_log_dir,
        ]
        subprocess.run(command, check=True)


def plot_roc(experiment_dir: str):
    subprocess.run([sys.executable, "-m", "analysis.roc", experiment_dir])


def plot_calib(experiment_dir: str):
    subprocess.run([sys.executable, "-m", "analysis.calib", experiment_dir])


def plot_purpose(experiment_dir: str, experiment_dir_open: str):
    subprocess.run([sys.executable, "-m", "analysis.purpose", experiment_dir, experiment_dir_open])


def plot_cot(experiment_dir: str):
    subprocess.run([sys.executable, "-m", "analysis.cot", experiment_dir])


def print_table(experiment_dir: str):
    subprocess.run([sys.executable, "-m", "analysis.score_table", experiment_dir])


def run_trace_local(limit: int, mini_suffix: str, model_id: str) -> None:
    """
    Run a local HuggingFace model telemetry pass over eval_awareness_val.json.
    Outputs go to logs/trace_{model_slug}{mini_suffix}/ and contain:
      records.parquet, activations/*.pt, manifest.json

    Works with any HF causal LM using the standard decoder architecture
    (model.model.layers[n]), e.g. Mistral, LLaMA, Gemma, Qwen.
    """
    from tqdm import tqdm
    from trace.dataset import load_trace_dataset
    from trace.windower import messages_to_windows
    from trace.tracer import load_model, unload_model, trace_sample
    from trace.writer import write_outputs

    device = "cuda"
    model_slug = model_id.split("/")[-1]
    output_dir = f"logs/trace_{model_slug}{mini_suffix}"
    data_limit = limit if limit < 1_000_000 else None

    records = load_trace_dataset(MAIN_DATA_FILE, limit=data_limit)
    print(f"Loaded {len(records)} samples  ({sum(r.label == 'EVAL' for r in records)} EVAL, "
          f"{sum(r.label == 'HARD_NEG' for r in records)} HARD_NEG, "
          f"{sum(r.label == 'REAL' for r in records)} REAL)")

    print(f"Loading model {model_id} ...")
    tokenizer, model = load_model(model_id=model_id, device=device)

    traces = []
    for rec in tqdm(records, desc="Tracing", unit="sample"):
        windows = messages_to_windows(rec.messages, tokenizer)
        trace = trace_sample(rec.sample_id, windows, model, device=device)
        traces.append(trace)

    print("Writing outputs ...")
    unload_model(model)
    write_outputs(traces, records, output_dir, model_id=model_id, limit=data_limit, device=device)
    print(f"Done → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run the dataset pipeline.")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset (just use a few samples for pipecleaning)")
    parser.add_argument("--full", action="store_true", help="Use full dataset")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", *STAGES],
        help="Stage to run"
    )
    parser.add_argument("--model", type=int, help="Model config to run")
    parser.add_argument(
        "--trace-model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model ID for the trace_local stage (any standard decoder LM)",
    )
    args = parser.parse_args()

    assert args.mini or args.full, "You must specify either --mini or --full"

    if args.stage == "all":
        stages = STAGES
    else:
        stages = [args.stage]

    if args.model is None:
        model_ixs = list(range(len(MODELS)))
    else:
        model_ixs = [args.model]

    if args.full:
        limit = 1_000_000
        mini_suffix = ""
        epochs = 5
    else:
        limit = 50
        mini_suffix = "_mini"
        epochs = 1

    for stage in stages:
        match stage:
            case "eval_mcq":
                eval_or_purpose_mcq(model_ixs, purpose=False, openended=False, limit=limit, experiment_dir=f"eval_mcq{mini_suffix}", epochs=epochs)
            case "purpose_mcq":
                eval_or_purpose_mcq(model_ixs, purpose=True, openended=False, limit=limit, experiment_dir=f"purpose_mcq{mini_suffix}", epochs=epochs)
            case "purpose_open":
                eval_or_purpose_mcq(model_ixs, purpose=True, openended=True, limit=limit, experiment_dir=f"purpose_open{mini_suffix}", epochs=epochs)
            case "eval_mcq_cot":
                eval_mcq_cot(model_ixs, input_experiment_dir=f"eval_mcq{mini_suffix}", output_experiment_dir=f"eval_mcq_cot{mini_suffix}", scorer="reasoning_classifier")
            case "purpose_open_cot":
                eval_mcq_cot(model_ixs, input_experiment_dir=f"purpose_open{mini_suffix}", output_experiment_dir=f"purpose_open_cot{mini_suffix}", scorer="openended_classifier")
            case "plot_roc":
                plot_roc(f"eval_mcq{mini_suffix}")
            case "plot_calib":
                plot_calib(f"eval_mcq{mini_suffix}")
            case "plot_purpose":
                plot_purpose(f"purpose_mcq{mini_suffix}", f"purpose_open_cot{mini_suffix}")
            case "plot_cot":
                plot_cot(f"eval_mcq_cot{mini_suffix}")
            case "print_table":
                print_table(f"eval_mcq{mini_suffix}")
            case "trace_local":
                run_trace_local(limit=limit, mini_suffix=mini_suffix, model_id=args.trace_model)
            case "probe_cheap":
                model_slug = args.trace_model.split("/")[-1]
                subprocess.run([sys.executable, "-m", "analysis.probe_cheap",
                                f"logs/trace_{model_slug}{mini_suffix}"], check=True)
            case "probe_acts":
                model_slug = args.trace_model.split("/")[-1]
                subprocess.run([sys.executable, "-m", "analysis.probe_acts",
                                f"logs/trace_{model_slug}{mini_suffix}"], check=True)
            case _:
                raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
