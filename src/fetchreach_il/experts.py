from __future__ import annotations

import numpy as np
import pygame

from .config import ACTION_DIM, DEFAULT_STEP_SIZE


class PIDController:
    def __init__(self, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.1, action_dim: int = 3, dt: float = 0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.action_dim = action_dim
        self.integral = np.zeros(action_dim, dtype=np.float32)
        self.prev_error = np.zeros(action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.integral = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_error = np.zeros(self.action_dim, dtype=np.float32)

    def compute(self, target_pos: np.ndarray, current_pos: np.ndarray) -> np.ndarray:
        error = target_pos - current_pos
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        action = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return action


def pid_expert_action(obs: dict, controller: PIDController, action_dim: int = ACTION_DIM) -> np.ndarray:
    ee_pos = np.asarray(obs["achieved_goal"], dtype=np.float32)
    target_pos = np.asarray(obs["desired_goal"], dtype=np.float32)
    action_xyz = controller.compute(target_pos, ee_pos)

    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = action_xyz
    action[3] = 0.0
    return np.clip(action, -1.0, 1.0)


def keyboard_expert_action(action_dim: int = ACTION_DIM, step_size: float = DEFAULT_STEP_SIZE) -> np.ndarray:
    action = np.zeros(action_dim, dtype=np.float32)
    valid_input = False

    while not valid_input:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            action[0] = -step_size
            valid_input = True
        elif keys[pygame.K_RIGHT]:
            action[0] = step_size
            valid_input = True

        if keys[pygame.K_UP]:
            action[1] = step_size
            valid_input = True
        elif keys[pygame.K_DOWN]:
            action[1] = -step_size
            valid_input = True

        if keys[pygame.K_w]:
            action[2] = -step_size
            valid_input = True
        elif keys[pygame.K_a]:
            action[2] = step_size
            valid_input = True

        if keys[pygame.K_q]:
            raise KeyboardInterrupt("User requested exit with Q")

    return action
