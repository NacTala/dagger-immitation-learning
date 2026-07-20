from __future__ import annotations

import os
import pickle
import time

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from fetchreach_il.config import ACTION_DIM, DATA_DIR, DEFAULT_STEP_SIZE, FETCHREACH_ENV_ID, MAX_EPISODE_STEPS, MANUAL_DATASET_PATH

ENV_ID = FETCHREACH_ENV_ID
SAVE_EVERY = 500
CV2_WINDOW = "FetchReach - Manual Control"
FPS = 60
REWARD_THRESHOLD = -122


def load_existing_buffers(out_path):
    if out_path.exists():
        with out_path.open("rb") as file:
            data = pickle.load(file)
        return data.get("obs", []), data.get("actions", []), data.get("infos", [])
    return [], [], []


def main() -> None:
    import cv2
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import numpy as np
    import pygame

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MANUAL_DATASET_PATH
    buf_obs, buf_actions, buf_infos = load_existing_buffers(out_path)

    print("=== Collecte manuelle pour DAgger - FetchReach ===")
    print("Contrôles :")
    print("  Flèche gauche / droite : axe X")
    print("  Flèche haut / bas : axe Y")
    print("  W / A : axe Z")
    print("  Q : quitter et sauvegarder")
    print()

    env = gym.make(ENV_ID, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
    obs, info = env.reset()
    action = np.zeros(ACTION_DIM, dtype=np.float32)

    pygame.init()
    pygame.display.set_mode((200, 100))
    pygame.display.set_caption("Contrôle manuel du robot")
    clock = pygame.time.Clock()
    cv2.namedWindow(CV2_WINDOW, cv2.WINDOW_NORMAL)

    episode_obs, episode_actions, episode_infos = [], [], []
    episode_reward = 0.0
    step_count = 0
    episode_count = 0
    running = True

    try:
        while running:
            action[:] = 0.0
            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                action[0] = -DEFAULT_STEP_SIZE
            if keys[pygame.K_RIGHT]:
                action[0] = DEFAULT_STEP_SIZE
            if keys[pygame.K_UP]:
                action[1] = DEFAULT_STEP_SIZE
            if keys[pygame.K_DOWN]:
                action[1] = -DEFAULT_STEP_SIZE
            if keys[pygame.K_w]:
                action[2] = -DEFAULT_STEP_SIZE
            if keys[pygame.K_a]:
                action[2] = DEFAULT_STEP_SIZE
            if keys[pygame.K_q]:
                running = False
                break

            if np.any(action != 0.0):
                action_to_env = np.clip(action, env.action_space.low, env.action_space.high)
                next_obs, reward, terminated, truncated, info = env.step(action_to_env)
                episode_reward += reward
                done_episode = terminated or truncated

                episode_obs.append(obs)
                episode_actions.append(action_to_env.copy())
                episode_infos.append(info)

                step_count += 1
                obs = next_obs

                if done_episode:
                    episode_count += 1
                    print(f"[Episode {episode_count}] terminé après {len(episode_actions)} étapes. Reward = {episode_reward}")

                    if episode_reward >= REWARD_THRESHOLD:
                        buf_obs.extend(episode_obs)
                        buf_actions.extend(episode_actions)
                        buf_infos.extend(episode_infos)
                        print("=> Episode sauvegardé")

                    episode_obs, episode_actions, episode_infos = [], [], []
                    episode_reward = 0.0
                    obs, info = env.reset()

            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(CV2_WINDOW, frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False
                break

            if step_count % SAVE_EVERY == 0 and step_count > 0:
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                with tmp_path.open("wb") as file:
                    pickle.dump({"obs": buf_obs, "actions": buf_actions, "infos": buf_infos}, file)
                os.replace(tmp_path, out_path)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sauvegarde intermédiaire -> {out_path}")

            pygame.event.pump()
            clock.tick(FPS)

    except KeyboardInterrupt:
        print("Interruption clavier reçue. Sauvegarde et sortie...")

    finally:
        print(f"Sauvegarde finale -> {out_path} (total steps collected = {len(buf_actions)})")
        with out_path.open("wb") as file:
            pickle.dump({"obs": buf_obs, "actions": buf_actions, "infos": buf_infos}, file)

        env.close()
        pygame.quit()
        cv2.destroyAllWindows()
        print("Terminé.")


if __name__ == "__main__":
    main()
