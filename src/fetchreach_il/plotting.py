from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import PLOTS_DIR


def plot_iteration_metrics(csv_path: Path | str, output_dir: Path | str | None = None) -> Path:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir is not None else PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df["iteration"], df["reward_cum"], marker="o", label="reward_cum")
    if "reward_std" in df.columns:
        axes[0].fill_between(df["iteration"], df["reward_cum"] - df["reward_std"], df["reward_cum"] + df["reward_std"], alpha=0.2)
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["iteration"], df["final_distance"], marker="o", color="tab:orange", label="final_distance")
    if "final_distance_std" in df.columns:
        axes[1].fill_between(
            df["iteration"],
            df["final_distance"] - df["final_distance_std"],
            df["final_distance"] + df["final_distance_std"],
            alpha=0.2,
            color="tab:orange",
        )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Final distance")
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(csv_path.stem)
    figure.tight_layout()

    output_path = output_dir / f"{csv_path.stem}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_sac_metrics(csv_path: Path | str, output_dir: Path | str | None = None) -> Path:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir is not None else PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(df["train_timesteps"], df["success_rate"], marker="o", color="tab:green")
    axes[0].set_ylabel("Success rate")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["train_timesteps"], df["final_distance_mean"], marker="o", color="tab:orange")
    axes[1].set_ylabel("Final distance")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["train_timesteps"], df["cost_mean"], marker="o", color="tab:red")
    axes[2].set_xlabel("Train timesteps")
    axes[2].set_ylabel("Cost")
    axes[2].grid(True, alpha=0.3)

    figure.suptitle(csv_path.stem)
    figure.tight_layout()

    output_path = output_dir / f"{csv_path.stem}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_metrics_file(csv_path: Path | str, output_dir: Path | str | None = None) -> Path:
    csv_path = Path(csv_path)
    columns = set(pd.read_csv(csv_path, nrows=1).columns)
    if "train_timesteps" in columns:
        return plot_sac_metrics(csv_path, output_dir=output_dir)
    if "iteration" in columns and "final_distance" in columns:
        return plot_iteration_metrics(csv_path, output_dir=output_dir)
    raise ValueError(f"Unsupported metrics format: {csv_path}")
