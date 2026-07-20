from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import torch
from stable_baselines3 import SAC
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    ACTION_DIM,
    DEFAULT_BC_EPOCHS,
    DEFAULT_DAGGER_FIT_EPOCHS,
    DEFAULT_DAGGER_ITERS,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_SAC_TIMESTEPS,
    FETCHREACH_ENV_ID,
    MAX_EPISODE_STEPS,
)
from .evaluation import evaluate_policy_metrics, evaluate_sac_model


def _ensure_parent_dir(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fit_policy(policy, obs: np.ndarray, acts: np.ndarray, n_epochs: int = 20, batch_size: int = 256, lr: float = 1e-3) -> float:
    dataset = TensorDataset(
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor(acts, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    last_loss = 0.0

    policy.train()
    for _ in range(n_epochs):
        losses = []
        for ob_batch, ac_batch in loader:
            pred = policy(ob_batch)
            loss = loss_fn(pred, ac_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        last_loss = float(np.mean(losses)) if losses else 0.0

    return last_loss


def train_behavioral_cloning(
    policy,
    obs: np.ndarray,
    acts: np.ndarray,
    env=None,
    metrics_path: Path | str | None = None,
    n_epochs: int = DEFAULT_BC_EPOCHS,
    batch_size: int = 256,
    lr: float = 1e-4,
    eval_every: int = 1,
) -> None:
    if metrics_path is not None:
        metrics_path = _ensure_parent_dir(metrics_path)
        with Path(metrics_path).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", "beta", "reward_cum", "reward_std", "final_distance", "final_distance_std"])

    for epoch in range(1, n_epochs + 1):
        loss = fit_policy(policy, obs, acts, n_epochs=1, batch_size=batch_size, lr=lr)
        print(f"[Epoch {epoch}/{n_epochs}] Loss={loss:.6f}")

        if env is not None and metrics_path is not None and epoch % eval_every == 0:
            metrics = evaluate_policy_metrics(policy, env, n_episodes=DEFAULT_EVAL_EPISODES)
            with Path(metrics_path).open("a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch,
                    1.0,
                    metrics["reward_mean"],
                    metrics["reward_std"],
                    metrics["dist_mean"],
                    metrics["dist_std"],
                ])


def _predict_action(policy, state: np.ndarray) -> np.ndarray:
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return policy(state_t).cpu().numpy()[0]


def collect_dagger_trajectory(
    env,
    policy,
    beta: float,
    expert_action_fn: Callable[[dict], np.ndarray],
    expert_reset_fn: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if expert_reset_fn is not None:
        expert_reset_fn()

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []

    obs, _ = env.reset()
    done = False

    while not done:
        state = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
        expert_action = np.asarray(expert_action_fn(obs), dtype=np.float32)
        policy_action = np.asarray(_predict_action(policy, state), dtype=np.float32)

        if np.random.rand() < beta:
            action_to_env = expert_action
        else:
            action_to_env = policy_action

        obs_list.append(state)
        act_list.append(expert_action)

        obs, reward, terminated, truncated, info = env.step(action_to_env)
        done = terminated or truncated

    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)


def train_dagger(
    policy,
    env,
    initial_obs: np.ndarray,
    initial_actions: np.ndarray,
    expert_action_fn: Callable[[dict], np.ndarray],
    expert_reset_fn: Callable[[], None] | None = None,
    metrics_path: Path | str | None = None,
    n_iterations: int = DEFAULT_DAGGER_ITERS,
    bc_epochs: int = DEFAULT_BC_EPOCHS,
    finetune_epochs: int = DEFAULT_DAGGER_FIT_EPOCHS,
    batch_size: int = 256,
    lr: float = 1e-3,
    eval_episodes: int = 100,
    beta_schedule: Callable[[int], float] | None = None,
) -> None:
    if metrics_path is not None:
        metrics_path = _ensure_parent_dir(metrics_path)
        with Path(metrics_path).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", "beta", "reward_cum", "reward_std", "final_distance", "final_distance_std"])

    dataset_obs = np.asarray(initial_obs, dtype=np.float32).copy()
    dataset_actions = np.asarray(initial_actions, dtype=np.float32).copy()

    print("\n=== Initial Behavioral Cloning ===")
    fit_policy(policy, dataset_obs, dataset_actions, n_epochs=bc_epochs, batch_size=batch_size, lr=lr)

    if beta_schedule is None:
        beta_schedule = lambda iteration: max(0.0, 1.0 - (iteration - 1) / n_iterations)

    print("\n=========== DAGGER TRAINING LOOP ===========")
    for iteration in range(1, n_iterations + 1):
        beta = float(beta_schedule(iteration))
        print(f"\n>>> Iteration {iteration}/{n_iterations} - beta = {beta:.2f}")

        new_obs, new_actions = collect_dagger_trajectory(env, policy, beta, expert_action_fn, expert_reset_fn)
        dataset_obs = np.concatenate([dataset_obs, new_obs], axis=0)
        dataset_actions = np.concatenate([dataset_actions, new_actions], axis=0)
        print(f"    Dataset size becomes: {dataset_obs.shape[0]} samples")

        print("    Fine-tuning policy...")
        loss = fit_policy(policy, dataset_obs, dataset_actions, n_epochs=finetune_epochs, batch_size=batch_size, lr=lr)
        print(f"    Fine-tuning loss: {loss:.6f}")

        eval_env = gym.make(FETCHREACH_ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
        metrics = evaluate_policy_metrics(policy, eval_env, n_episodes=eval_episodes)
        eval_env.close()

        if metrics_path is not None:
            with Path(metrics_path).open("a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    iteration,
                    beta,
                    metrics["reward_mean"],
                    metrics["reward_std"],
                    metrics["dist_mean"],
                    metrics["dist_std"],
                ])


def train_sac_schedule(
    timesteps_list: list[int] | None = None,
    metrics_path: Path | str | None = None,
    model_dir: Path | str | None = None,
    env_id: str = FETCHREACH_ENV_ID,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    sac_kwargs: dict | None = None,
) -> list[dict[str, float]]:
    timesteps_list = timesteps_list or list(DEFAULT_SAC_TIMESTEPS)
    sac_kwargs = sac_kwargs or {}
    model_dir = Path(model_dir) if model_dir is not None else None
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float]] = []

    if metrics_path is not None:
        metrics_path = _ensure_parent_dir(metrics_path)
        with Path(metrics_path).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["train_timesteps", "success_rate", "final_distance_mean", "cost_mean", "training_time_seconds"])

    for timesteps in timesteps_list:
        print(f"\n=== Training SAC for {timesteps} timesteps ===")
        env = gym.make(env_id, render_mode=None)
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=0,
            learning_rate=1e-3,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            **sac_kwargs,
        )

        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        training_time = time.time() - start_time

        if model_dir is not None:
            model_name = model_dir / f"sac_fetchreach_{timesteps}steps"
            model.save(str(model_name))
            print(f"Model saved to {model_name}.zip")

        eval_env = gym.make(env_id)
        metrics = evaluate_sac_model(model, eval_env, n_episodes=eval_episodes)
        eval_env.close()
        env.close()

        row = {
            "train_timesteps": float(timesteps),
            "success_rate": metrics["success_rate"],
            "final_distance_mean": metrics["final_distance_mean"],
            "cost_mean": metrics["cost_mean"],
            "training_time_seconds": float(training_time),
        }
        results.append(row)

        if metrics_path is not None:
            with Path(metrics_path).open("a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    timesteps,
                    metrics["success_rate"],
                    metrics["final_distance_mean"],
                    metrics["cost_mean"],
                    training_time,
                ])

    return results
