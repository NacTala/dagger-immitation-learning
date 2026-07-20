from __future__ import annotations

import numpy as np
import torch


def _policy_action(policy, state: np.ndarray) -> np.ndarray:
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = policy(state_t)
        if hasattr(action, "detach"):
            action = action.detach()
        return action.cpu().numpy()[0]


def evaluate_policy_metrics(policy, env, n_episodes: int = 20) -> dict[str, float]:
    rewards = []
    final_distances = []
    success_count = 0

    if hasattr(policy, "eval"):
        policy.eval()

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state = np.concatenate([obs["observation"], obs["desired_goal"]])
            action = _policy_action(policy, state)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
        final_distances.append(float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])))
        success_count += int(float(info.get("is_success", 0.0)) == 1.0)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "dist_mean": float(np.mean(final_distances)),
        "dist_std": float(np.std(final_distances)),
        "success_rate": float(success_count / n_episodes),
    }


def evaluate_sac_model(model, env, n_episodes: int = 20) -> dict[str, float]:
    success_count = 0
    final_distances = []
    costs = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_cost = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_cost += float(np.linalg.norm(obs["observation"][0:3] - obs["desired_goal"]))

        success_count += int(float(info.get("is_success", 0.0)) == 1.0)
        final_distances.append(float(np.linalg.norm(obs["observation"][0:3] - obs["desired_goal"])))
        costs.append(ep_cost)

    return {
        "success_rate": float(success_count / n_episodes),
        "final_distance_mean": float(np.mean(final_distances)),
        "final_distance_std": float(np.std(final_distances)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
    }
