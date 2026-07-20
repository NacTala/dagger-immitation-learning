from __future__ import annotations

import argparse

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from fetchreach_il.config import DEFAULT_SAC_TIMESTEPS, METRICS_DIR, MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on FetchReach.")
    parser.add_argument("--timesteps", type=int, nargs="*", default=DEFAULT_SAC_TIMESTEPS, help="Training schedule.")
    parser.add_argument("--metrics-path", default=str(METRICS_DIR / "sac_metrics.csv"), help="CSV file to save metrics.")
    parser.add_argument("--model-dir", default=str(MODELS_DIR), help="Directory to save trained SAC models.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of evaluation episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from fetchreach_il.training import train_sac_schedule

    train_sac_schedule(
        timesteps_list=args.timesteps,
        metrics_path=args.metrics_path,
        model_dir=args.model_dir,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":
    main()
