import json
import os

MAIN_DATA_FILE = "eval_awareness/dataset.json"

def get_categories():
    with open(MAIN_DATA_FILE) as f:
        data = json.load(f)
    categories = list(set([item["metadata"]["category"] for item in data]))
    agent_categories = list(
        set(
            [
                item["metadata"]["category"]
                for item in data
                if item["metadata"]["scaffold"] is not None
            ]
        )
    )
    eval_categories = list(
        set(
            [
                item["metadata"]["category"]
                for item in data
                if item["metadata"]["eval_category"]
            ]
        )
    )
    return categories, agent_categories, eval_categories


CAPABILITY_EVAL_CATEGORIES = [
    "cybench",
    "swe_gym",
    "swebench",
    "nl2bash",
    "gaia_runs",
    "mmlu",
    "aider_full",
    "natural_reasoning",
    "smol_gaia_manager",
    "smol_gaia_search",
    "wmdp",
]

_categories_cache: dict | None = None


def _ensure_categories() -> dict:
    global _categories_cache
    if _categories_cache is None:
        all_cats, agent_cats, eval_cats = get_categories()
        _categories_cache = {
            "ALL_CATEGORIES": all_cats,
            "AGENT_CATEGORIES": agent_cats,
            "EVAL_CATEGORIES": eval_cats,
            "REAL_CATEGORIES": [c for c in all_cats if c not in eval_cats],
            "ALIGNMENT_EVAL_CATEGORIES": [c for c in eval_cats if c not in CAPABILITY_EVAL_CATEGORIES],
        }
    return _categories_cache


def __getattr__(name: str):
    lazy = {"ALL_CATEGORIES", "AGENT_CATEGORIES", "EVAL_CATEGORIES", "REAL_CATEGORIES", "ALIGNMENT_EVAL_CATEGORIES"}
    if name in lazy:
        return _ensure_categories()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def find_most_recent_file(path: str) -> str | None:
    choices = []
    if not os.path.exists(path):
        return None
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".eval"):
                choices.append(entry.path)
    if len(choices) == 0:
        return None
    choices.sort()
    return choices[-1]  # Return the most recent file
