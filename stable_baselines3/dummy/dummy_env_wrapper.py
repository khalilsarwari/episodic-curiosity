# coding=utf-8
# Copyright 2019 Google LLC.
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

from stable_baselines3.reachability import episodic_memory
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper
import gym
import numpy as np
import cv2
import torch

class DummyEnvWrapper(VecEnvWrapper):
  """Environment wrapper that adds additional curiosity reward."""

  def __init__(self,
               vec_env,
               add_stoch):

    self.add_stoch = add_stoch
    observation_space = vec_env.observation_space

    VecEnvWrapper.__init__(self, vec_env, observation_space=observation_space)

  def step_wait(self):
    """Overrides VecEnvWrapper.step_wait."""
    observations, rewards, dones, infos = self.venv.step_wait()
    if self.add_stoch:
      noise = np.random.randint(low=-10, high=10, size=observations.shape)
      noisy_obs = observations + noise
      observations = np.clip(noisy_obs, 0, 255)

    return observations, rewards, dones, infos

  def reset(self):
    """Overrides VecEnvWrapper.reset."""
    observations = self.venv.reset()

    return observations