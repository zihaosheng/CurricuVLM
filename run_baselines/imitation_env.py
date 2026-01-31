# Copyright 2024 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Environments interface of SafeMetaDrive."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
import torch

META_DRIVE_AVAILABLE = True
try:
    from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
except ImportError:
    META_DRIVE_AVAILABLE = False

from gymnasium import spaces

discrete_actions = {
    0: [-1, 1], 1: [0, 1], 2: [1, 1], 3: [0, 0],
}

def convert_gym_to_gymnasium(old_box):
    low = old_box.low
    high = old_box.high
    shape = old_box.shape
    dtype = old_box.dtype

    new_box = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return new_box


class ImitationMetaDriveEnv(gym.Env):

    def __init__(
        self,
        num_envs: int = 1,
        device: torch.device = "cpu",
        config: dict = None,
        action_space_type="continuous",
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> None:
        """Initialize an instance of :class:`SafetyMetaDriveEnv`."""
        super().__init__()
        self._num_envs = num_envs
        self._device = torch.device(device)

        if META_DRIVE_AVAILABLE:
            self._env = WaymoEnv(config=config)
        else:
            raise ImportError(
                'Please install MetaDrive to use SafeMetaDrive!\
                \nInstall from source: https://github.com/metadriverse/metadrive.\
                \nInstall from PyPI: `pip install metadrive`.',
            )
        self._num_scenarios = self._env.config['num_scenarios']
        self.last_seed = 0

        # self._env.logger.setLevel(logging.FATAL)
        self.action_space_type = action_space_type
        if self.action_space_type == "continuous":
            self.action_space = convert_gym_to_gymnasium(self._env.action_space)
        elif self.action_space_type == "discrete":
            self.action_space = gym.spaces.Discrete(len(discrete_actions))
        self.observation_space = convert_gym_to_gymnasium(self._env.observation_space)
        print(self.action_space)
        print(type(self.action_space))
        self._metadata = self._env.metadata

    def step(self, action):
        if self.action_space_type == "discrete":
            action = discrete_actions[action]
        obs, reward, done, info = self._env.step(
            action,
        )
        terminated = done
        truncated = done
        return obs, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: Optional[Dict[str, Any]] = None):
        obs = self._env.reset(force_seed=seed)
        return obs, {}

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed)

    def render(self, *args, **kwargs) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            The render frames: we recommend to use `np.ndarray`
                which could construct video by moviepy.
        """
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    @property
    def current_seed(self):
        return self._env.current_seed

    @property
    def engine(self):
        return self._env.engine

    @property
    def vehicle(self):
        return self._env.vehicle
