from matplotlib import colormaps

tab10 = colormaps['tab10'].colors

# If you change these, let me know so I can regenerate the plots
MODEL_COLORS = {
    "claude-3-5-sonnet-20241022":tab10[5], # brown
    "claude-3-7-sonnet-20250219":tab10[1], # orange
    "claude-3-7-sonnet-20250219-16384":tab10[3], # red
    "gpt-4o-2024-11-20":tab10[4],  # purple
    "gpt-4.1-2025-04-14":tab10[0], # blue
    "gpt-4.1-mini-2025-04-14":tab10[9], # light blue
    "o4-mini-2025-04-16-medium":tab10[6], # pink
    "gemini-2.5-pro-preview-03-25": tab10[2], # green
    "gemini-2.5-flash-preview": tab10[8], # olive
    "Mistral-7B-Instruct-v0.3": tab10[7], # gray
}

MODEL_NICE_NAMES = {
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-3-7-sonnet-20250219-16384": "Claude 3.7 Sonnet*",
    "gpt-4o-2024-11-20": "GPT-4o",
    "gpt-4.1-2025-04-14": "GPT-4.1",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 mini",
    "o4-mini-2025-04-16-medium": "o4-mini*",
    "gemini-2.5-pro-preview-03-25": "Gemini 2.5 Pro*",
    "gemini-2.5-flash-preview": "Gemini 2.5 Flash",
    "Mistral-7B-Instruct-v0.3": "Mistral 7B",
}


EVAL_PALETTE = [tab10[0], tab10[1]]