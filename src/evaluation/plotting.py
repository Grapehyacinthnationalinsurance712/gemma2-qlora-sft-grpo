"""
Training curve and reward visualisation utilities.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


EXP_STYLES = [
    {"color": "#4C72B0", "ls": "-",  "marker": "o"},
    {"color": "#DD8452", "ls": "--", "marker": "s"},
    {"color": "#55A868", "ls": "-.", "marker": "^"},
    {"color": "#C44E52", "ls": ":",  "marker": "D"},
    {"color": "#8172B3", "ls": "-",  "marker": "P"},
    {"color": "#937860", "ls": "--", "marker": "X"},
]


def _smooth(values: List[float], window: int = 5) -> List[float]:
    """Apply a simple moving average to a list of values."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i + 1]) / (i - start + 1))
    return out


def _short_label(key: str) -> str:
    """Convert a W&B reward key to a short display label.

    Example: 'rewards/correctness_reward_fn/mean' → 'Correctness'
    """
    parts = key.replace("train/", "").split("/")
    name = parts[1] if len(parts) > 1 else key
    return name.replace("_reward_fn", "").replace("_fn", "").replace("_", " ").title()


def plot_training_loss(
    log_history: List[Dict],
    title: str = "Training Loss",
    smooth_window: int = 5,
) -> None:
    """Plot training loss from a W&B / HuggingFace log history.

    Args:
        log_history: List of dicts loaded from trainer_state.json or log_history.json.
        title: Plot title.
        smooth_window: Moving average window size.
    """
    steps = [e["step"] for e in log_history if "step" in e and "loss" in e]
    losses = [e["loss"] for e in log_history if "step" in e and "loss" in e]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, alpha=0.2, color="#4C72B0", lw=1)
    ax.plot(steps, _smooth(losses, smooth_window), color="#4C72B0", lw=2,
            label=f"Loss (smoothed, w={smooth_window})")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_reward_curves(
    log_history: List[Dict],
    reward_keys: Optional[List[str]] = None,
    title: str = "Reward Components vs. Steps",
    smooth_window: int = 5,
) -> None:
    """Plot individual reward components from a GRPO training log.

    Args:
        log_history: List of dicts from log_history.json.
        reward_keys: List of reward key names to plot. Auto-detected if None.
        title: Plot title.
        smooth_window: Moving average window size.
    """
    if reward_keys is None:
        all_keys = set()
        for e in log_history:
            all_keys.update(e.keys())
        reward_keys = sorted(
            k for k in all_keys if "reward" in k.lower() and "std" not in k.lower()
        )

    if not reward_keys:
        print("No reward keys found in log history.")
        return

    fig, axes = plt.subplots(1, len(reward_keys), figsize=(5 * len(reward_keys), 4))
    if len(reward_keys) == 1:
        axes = [axes]

    for ax, rkey in zip(axes, reward_keys):
        rows = [(e["step"], e[rkey]) for e in log_history if "step" in e and rkey in e]
        if not rows:
            continue
        steps, vals = zip(*rows)
        ax.plot(steps, vals, alpha=0.2, color="#DD8452", lw=1)
        ax.plot(steps, _smooth(list(vals), smooth_window), color="#DD8452", lw=2)

        std_key = rkey.replace("/mean", "/std")
        if std_key in (log_history[0] if log_history else {}):
            stds = [e.get(std_key, 0) for e in log_history if "step" in e and rkey in e]
            m = list(vals)
            ax.fill_between(steps,
                            [m[i] - stds[i] for i in range(len(m))],
                            [m[i] + stds[i] for i in range(len(m))],
                            alpha=0.1, color="#DD8452")

        ax.set_title(_short_label(rkey), fontsize=11, weight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_experiment_comparison(
    exp_dir: Path,
    section: str,
    smooth_window: int = 5,
) -> Dict:
    """Overlay training curves for all experiments in a directory.

    Scans exp_dir for subfolders containing log_history.json and plots
    loss and reward components side by side for comparison.

    Args:
        exp_dir: Directory containing experiment subfolders.
        section: Label for the plot title.
        smooth_window: Moving average window size.

    Returns:
        Dict mapping experiment labels to their parsed log DataFrames.
    """
    subdirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    experiments = {}

    for folder in subdirs:
        log_path = folder / "log_history.json"
        if not log_path.exists():
            continue
        log = json.load(open(log_path))
        all_keys = set()
        for e in log:
            all_keys.update(e.keys())
        rk_mean = sorted(k for k in all_keys if "reward" in k.lower() and "std" not in k)
        rows = [
            {
                "step": e["step"],
                **{k: e[k] for k in (["loss"] + rk_mean) if k in e},
            }
            for e in log if "step" in e
        ]
        if rows:
            label = folder.name
            experiments[label] = {
                "df": pd.DataFrame(rows).sort_values("step").reset_index(drop=True),
                "reward_mean_keys": rk_mean,
            }

    if not experiments:
        print("No valid experiments found.")
        return {}

    all_reward_keys = list(dict.fromkeys(
        k for exp in experiments.values() for k in exp["reward_mean_keys"]
    ))
    n_rewards = max(len(all_reward_keys), 1)

    fig = plt.figure(figsize=(max(13, 5 * n_rewards), 8))
    gs = fig.add_gridspec(2, n_rewards, hspace=0.38, wspace=0.30, height_ratios=[1, 1])
    ax_loss = fig.add_subplot(gs[0, :])
    ax_rews = [fig.add_subplot(gs[1, j]) for j in range(n_rewards)]

    legend_handles = []
    for idx, (label, exp) in enumerate(experiments.items()):
        df = exp["df"]
        sty = EXP_STYLES[idx % len(EXP_STYLES)]
        c, ls, mk = sty["color"], sty["ls"], sty["marker"]

        if "loss" in df.columns:
            sub = df[["step", "loss"]].dropna()
            steps, vals = sub["step"].values, sub["loss"].values
            me = max(1, len(steps) // 8)
            ax_loss.plot(steps, vals, alpha=0.15, color=c, lw=1)
            h, = ax_loss.plot(
                steps, _smooth(vals.tolist(), smooth_window),
                label=label, color=c, ls=ls, lw=2, marker=mk,
                markersize=5, markevery=me,
            )
            legend_handles.append(h)

        for col_idx, rkey in enumerate(all_reward_keys):
            ax = ax_rews[col_idx]
            if rkey not in df.columns:
                continue
            sub = df[["step", rkey]].dropna()
            steps, vals = sub["step"].values, sub[rkey].values
            me = max(1, len(steps) // 8)
            ax.plot(steps, vals, alpha=0.15, color=c, lw=1)
            ax.plot(steps, _smooth(vals.tolist(), smooth_window),
                    color=c, ls=ls, lw=2, marker=mk, markersize=4, markevery=me)

    ax_loss.set_title(f"{section} — Training Loss (faint=raw, bold=smoothed)")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)

    for col_idx, rkey in enumerate(all_reward_keys):
        ax = ax_rews[col_idx]
        ax.set_title(_short_label(rkey))
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper center",
                   ncol=min(len(experiments), 5), fontsize=9,
                   bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{section} Experiment Comparison", fontsize=13, weight="bold", y=1.06)
    plt.tight_layout()
    plt.show()

    return experiments