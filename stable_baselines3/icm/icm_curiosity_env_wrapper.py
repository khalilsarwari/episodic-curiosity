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

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper
import gym
import numpy as np
import cv2

def resize_observation(frame, image_shape, reward=None):
  """Resize an observation according to the target image shape."""
  # Shapes already match, nothing to be done
  height, width, target_depth = image_shape
  if frame.shape == (height, width, target_depth):
    return frame
  if frame.shape[-1] != 3 and frame.shape[-1] != 1:
    raise ValueError(
        'Expecting color or grayscale images, got shape {}: {}'.format(
            frame.shape, frame))

  if frame.shape[-1] == 3 and target_depth == 1:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

  frame = cv2.resize(frame, (width, height),
                     interpolation=cv2.INTER_AREA)

  # OpenCV operations removes the last axis for grayscale images.
  # Restore the last axis.
  if len(frame.shape) != 3:
    frame = frame[:, :, np.newaxis]

  if reward is None:
    return frame

  return np.concatenate([frame, np.full([height, width, 1], reward)], axis=-1)


class MovingAverage(object):
  """Computes the moving average of a variable."""

  def __init__(self, capacity):
    self._capacity = capacity
    self._history = np.array([0.0] * capacity)
    self._size = 0

  def add(self, value):
    index = self._size % self._capacity
    self._history[index] = value
    self._size += 1

  def mean(self):
    if not self._size:
      return None
    if self._size < self._capacity:
      return np.mean(self._history[0:self._size])
    return np.mean(self._history)


class ICMCuriosityEnvWrapper(VecEnvWrapper):
  """Environment wrapper that adds additional curiosity reward."""

  def __init__(self,
               vec_env,
               reward_fn,
               forward_fn,
               target_image_shape,
               add_stoch,
               exploration_reward = 'intrinsic_curiosity',
               scale_task_reward = 1.0,
               scale_surrogate_reward = 0.5,
               append_ec_reward_as_channel = False,
               bonus_reward_additive_term = 0,
               exploration_reward_min_step = 5000):

    self.add_stoch = add_stoch
    # Note: post-processing of the observation might change the [0, 255]
    # range of the observation...
    if self._should_postprocess_observation(vec_env.observation_space.shape):
      observation_space_shape = target_image_shape[:]
      if append_ec_reward_as_channel:
        observation_space_shape[-1] += 1
      observation_space = gym.spaces.Box(
          low=0, high=255, shape=observation_space_shape, dtype=np.uint8)
    else:
      observation_space = vec_env.observation_space
      assert not append_ec_reward_as_channel, (
          'append_ec_reward_as_channel not compatible with non-image-like obs.')

    VecEnvWrapper.__init__(self, vec_env, observation_space=observation_space)

    self._bonus_reward_additive_term = bonus_reward_additive_term
    self._reward_fn = reward_fn
    self._forward_fn = forward_fn
    self._target_image_shape = target_image_shape
    self._append_ec_reward_as_channel = append_ec_reward_as_channel

    self._exploration_reward = exploration_reward
    self._scale_task_reward = scale_task_reward
    self._scale_surrogate_reward = scale_surrogate_reward
    self._exploration_reward_min_step = exploration_reward_min_step

    # Cumulative task reward over an episode.
    self._episode_task_reward = [0.0] * self.venv.num_envs
    self._episode_bonus_reward = [0.0] * self.venv.num_envs
    self._episode_uncertainty_reward = [0.0] * self.venv.num_envs

    # Stats on the task and exploration reward.
    self._stats_task_reward = MovingAverage(capacity=100)
    self._stats_bonus_reward = MovingAverage(capacity=100)
    self._stats_uncertainty_reward = MovingAverage(capacity=100)

    # Total number of steps so far per environment.
    self._step_count = 0

    # Observers are notified each time a new time step is generated by the
    # environment.
    # Observers implement a function "on_new_observation".
    self._observers = []

  def _should_postprocess_observation(self, obs_shape):
    # Only post-process observations that look like an image.
    return len(obs_shape) >= 3

  def add_observer(self, observer):
    self._observers.append(observer)

  def _postprocess_observation(self, observations, rewards=None):
    if not self._should_postprocess_observation(observations[0].shape):
      return observations

    if self._append_ec_reward_as_channel:
      if rewards is not None:
        return np.array(
            [resize_observation(obs, self._target_image_shape, rew)
             for obs, rew in zip(observations, rewards)])
      else:
        # When environment is reset there are no rewards, so we explicitly pass
        # 0 in this case.
        return np.array(
            [resize_observation(obs, self._target_image_shape, 0)
             for obs in observations])
    else:
      return np.array(
          [resize_observation(obs, self._target_image_shape, None)
           for obs in observations])

  def _compute_icm_reward(self, obs, actions, next_obs, rewards, dones, infos):
    for observer in self._observers:
      observer.on_new_observation(obs, actions, next_obs, rewards, dones, infos)

    if self._exploration_reward == 'intrinsic_curiosity':
      feature_rewards, uncertainty_rewards = self._reward_fn(obs, actions, next_obs)
      b_rewards = np.zeros(self.venv.num_envs)
      u_rewards = np.zeros(self.venv.num_envs)
      for k in range(self.venv.num_envs):
        b_rewards[k] = feature_rewards[k]
        u_rewards[k] = uncertainty_rewards[k]

      bonus_rewards = [
          0.0 if d else 0.5 + s + u + self._bonus_reward_additive_term
          for (s, u, d) in zip(b_rewards, u_rewards, dones)
      ]

      bonus_rewards = np.array(bonus_rewards)
    elif self._exploration_reward == 'none':
      bonus_rewards = np.zeros(self.venv.num_envs)
    else:
      raise ValueError('Unknown exploration reward: {}'.format(
          self._exploration_reward))

    # Combined rewards.
    scale_surrogate_reward = self._scale_surrogate_reward
    if self._step_count < self._exploration_reward_min_step:
      # This can be used for online training during the first N steps,
      # the ICM is totally random and the surrogate reward has no
      # meaning.
      scale_surrogate_reward = 0.0

    postprocessed_rewards = (self._scale_task_reward * rewards +
                             scale_surrogate_reward * bonus_rewards)

    # Update the statistics.
    for i in range(self.venv.num_envs):
      self._episode_task_reward[i] += rewards[i]
      self._episode_bonus_reward[i] += bonus_rewards[i]
      self._episode_uncertainty_reward[i] += u_rewards[i]
      if dones[i]:
        self._stats_task_reward.add(self._episode_task_reward[i])
        self._stats_bonus_reward.add(self._episode_bonus_reward[i])
        self._stats_uncertainty_reward.add(self._episode_uncertainty_reward[i])
        self._episode_task_reward[i] = 0.0
        self._episode_bonus_reward[i] = 0.0
        self._episode_uncertainty_reward[i] = 0.0

    postprocessed_rewards = np.array(postprocessed_rewards)
    return postprocessed_rewards

  def step_wait(self):
    """Overrides VecEnvWrapper.step_wait."""
    observations, rewards, dones, infos = self.venv.step_wait()
    if self.add_stoch:
      noise = np.random.normal(loc=0, scale= 255**2,size=observations.shape)
      noisy_obs = observations.astype(np.float32) + noise
      observations = np.clip(noisy_obs.astype(np.uint8), 0, 255)

    self._step_count += 1

    if (self._step_count % 1000) == 0:
      print('step={} task_reward={} bonus_reward={} scale_bonus={}'.format(
          self._step_count,
          self._stats_task_reward.mean(),
          self._stats_bonus_reward.mean(),
          self._scale_surrogate_reward))

    for i in range(self.venv.num_envs):
      infos[i]['task_reward'] = rewards[i]
      infos[i]['task_observation'] = observations[i]

    postprocessed_rewards = rewards

    # Post-processing on the observation. Note that the reward could be used
    # as an input to the agent. For simplicity we add it as a separate channel.
    postprocessed_observations = self._postprocess_observation(observations,None)

    return postprocessed_observations, postprocessed_rewards, dones, infos

  def reset(self):
    """Overrides VecEnvWrapper.reset."""
    observations = self.venv.reset()
    postprocessed_observations = self._postprocess_observation(observations)

    return postprocessed_observations