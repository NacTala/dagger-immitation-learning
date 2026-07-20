from __future__ import annotations

import argparse
from pathlib import Path

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from fetchreach_il.config import (
    DEFAULT_BC_EPOCHS,
    FETCHREACH_ENV_ID,
    MAX_EPISODE_STEPS,
    MANUAL_DATASET_PATH,
    METRICS_DIR,
    MODELS_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavioral cloning on FetchReach data.")
    parser.add_argument("--dataset-path", default=str(MANUAL_DATASET_PATH), help="Path to the manual dataset pickle.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_BC_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs.")
    parser.add_argument("--metrics-path", default=str(METRICS_DIR / "bc_metrics.csv"), help="CSV file to save metrics.")
    parser.add_argument("--model-path", default=str(MODELS_DIR / "bc_policy.pt"), help="Path to save the trained policy state dict.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import torch

    from fetchreach_il.data import load_manual_dataset
    from fetchreach_il.models import BCPolicy
    from fetchreach_il.training import train_behavioral_cloning

    obs, acts = load_manual_dataset(args.dataset_path)

    env = gym.make(FETCHREACH_ENV_ID, render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    policy = BCPolicy(obs_dim=obs.shape[1], act_dim=acts.shape[1])

    train_behavioral_cloning(
        policy,
        obs,
        acts,
        env=env,
        metrics_path=args.metrics_path,
        n_epochs=args.epochs,
        eval_every=args.eval_every,
    )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), model_path)
    env.close()
    print(f"BC model saved to {model_path}")


if __name__ == "__main__":
    main()
