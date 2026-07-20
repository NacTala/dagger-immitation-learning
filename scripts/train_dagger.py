from __future__ import annotations

import argparse
from pathlib import Path

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from fetchreach_il.config import (
    DEFAULT_DAGGER_FIT_EPOCHS,
    DEFAULT_DAGGER_ITERS,
    DEFAULT_EVAL_EPISODES,
    FETCHREACH_ENV_ID,
    MAX_EPISODE_STEPS,
    MANUAL_DATASET_PATH,
    METRICS_DIR,
    MODELS_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DAgger on FetchReach.")
    parser.add_argument("--dataset-path", default=str(MANUAL_DATASET_PATH), help="Path to the initial demonstration dataset.")
    parser.add_argument("--expert", choices=["pid", "human", "sac"], default="pid", help="Expert used for aggregation.")
    parser.add_argument("--expert-model-path", default=str(MODELS_DIR / "sac_fetchreach_100000steps.zip"), help="Path to the SAC model used when expert=sac.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_DAGGER_ITERS, help="Number of DAgger iterations.")
    parser.add_argument("--bc-epochs", type=int, default=50, help="Initial behavioral cloning epochs.")
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_DAGGER_FIT_EPOCHS, help="Fine-tuning epochs per DAgger iteration.")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES, help="Episodes for validation after each iteration.")
    parser.add_argument("--metrics-path", default=str(METRICS_DIR / "dagger_metrics.csv"), help="CSV file to save DAgger metrics.")
    parser.add_argument("--model-path", default=str(MODELS_DIR / "dagger_policy.pt"), help="Path to save the trained policy state dict.")
    return parser.parse_args()


def build_expert(mode: str, env, expert_model_path: str):
    if mode == "pid":
        from fetchreach_il.experts import PIDController, pid_expert_action

        controller = PIDController(Kp=2.0, Ki=0.0, Kd=0.1)

        def expert_action(obs):
            return pid_expert_action(obs, controller)

        return expert_action, controller.reset

    if mode == "human":
        import pygame

        from fetchreach_il.experts import keyboard_expert_action

        pygame.init()
        pygame.display.set_mode((200, 100))
        pygame.display.set_caption("FetchReach manual expert")

        def expert_action(obs):
            return keyboard_expert_action()

        return expert_action, None

    from stable_baselines3 import SAC

    sac_model = SAC.load(str(expert_model_path), env=env)

    def expert_action(obs):
        action, _ = sac_model.predict(obs, deterministic=True)
        return action

    return expert_action, None


def main() -> None:
    args = parse_args()

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import pygame

    from fetchreach_il.data import load_manual_dataset
    from fetchreach_il.models import BCPolicy
    from fetchreach_il.training import train_dagger

    obs, acts = load_manual_dataset(args.dataset_path)
    env = gym.make(FETCHREACH_ENV_ID, render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    policy = BCPolicy(obs_dim=obs.shape[1], act_dim=acts.shape[1])

    expert_action_fn, expert_reset_fn = build_expert(args.expert, env, args.expert_model_path)

    try:
        train_dagger(
            policy,
            env,
            obs,
            acts,
            expert_action_fn=expert_action_fn,
            expert_reset_fn=expert_reset_fn,
            metrics_path=args.metrics_path,
            n_iterations=args.iterations,
            bc_epochs=args.bc_epochs,
            finetune_epochs=args.finetune_epochs,
            eval_episodes=args.eval_episodes,
        )
    finally:
        env.close()
        pygame.quit()

    import torch

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), model_path)
    print(f"DAgger policy saved to {model_path}")


if __name__ == "__main__":
    main()
