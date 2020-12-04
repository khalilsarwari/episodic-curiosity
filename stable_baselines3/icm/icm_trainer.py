"""Set of functions used to train an ICM"""

import os
import random

from torch import nn
import torch
import numpy as np
from tqdm import tqdm

LR = 1e-4
# TODO: put param above (LR) into a config and pass to class

class ICMTrainer(object):
  """Train an ICM network in an online way."""

  def __init__(self,
               icm_model,
               observation_history_size=20000,
               training_interval=20000,
               num_epochs=4,
               checkpoint_dir=None):
    # The training interval is assumed to be the same as the history size
    # for invalid negative values.
    if training_interval < 0:
      training_interval = observation_history_size

    self._icm_model = icm_model
    self._opt = torch.optim.Adam(icm_model.parameters(), lr=LR)

    self._training_interval = training_interval
    self._batch_size = 128
    self._num_epochs = num_epochs

    # Keeps track of the last N observations.
    # Those are used to train the ICM network in an online way.
    self._fifo_observations = [None] * observation_history_size
    self._fifo_actions = [None] * observation_history_size
    self._fifo_next_observations = [None] * observation_history_size
    self._fifo_dones = [None] * observation_history_size
    self._fifo_index = 0
    self._fifo_count = 0

    # Used to save checkpoints.
    self._current_epoch = 0
    self._checkpointer = None
    if checkpoint_dir is not None:
      checkpoint_period_in_epochs = self._num_epochs
      self._checkpointer = keras_checkpoint.GFileModelCheckpoint(
          os.path.join(checkpoint_dir, 'r_network_weights.{epoch:05d}.h5'),
          save_summary=False,
          save_weights_only=True,
          period=checkpoint_period_in_epochs)
      self._checkpointer.set_model(self._r_model)

  def on_new_observation(self, obs, actions, next_obs, unused_rewards, dones, infos):
    """Event triggered when the environments generate a new observation."""
    if len(obs.shape) >= 3 or infos is None or 'frame' not in infos:
      self._fifo_observations[self._fifo_index] = obs
    else:
      # Specific to Parkour (stores velocity, joints as the primary
      # observation).
      self._fifo_observations[self._fifo_index] = infos['frame']
    self._fifo_actions[self._fifo_index] = actions
    self._fifo_next_observations[self._fifo_index] = next_obs
    self._fifo_dones[self._fifo_index] = dones
    self._fifo_index = (
        (self._fifo_index + 1) % len(self._fifo_observations))
    self._fifo_count += 1

    if (self._fifo_count > 0 and
        self._fifo_count % self._training_interval == 0):
      print('Training the ICM after: {}'.format(self._fifo_count))
      history_observations, history_actions, history_next_observations, history_dones = self._get_flatten_history()
      self.train(history_observations, history_actions, history_next_observations, history_dones)

  def _get_flatten_history(self):
    """Convert the history given as a circular fifo to a linear array."""
    if self._fifo_count < len(self._fifo_observations):
      return (self._fifo_observations[:self._fifo_count],
      		  self._fifo_actions[:self._fifo_count],
      		  self._fifo_next_observations[:self._fifo_count],
              self._fifo_dones[:self._fifo_count])

    # Reorder the indices.
    history_observations = self._fifo_observations[self._fifo_index:]
    history_observations.extend(self._fifo_observations[:self._fifo_index])
    history_actions = self._fifo_actions[self._fifo_index:]
    history_actions.extend(self._fifo_actions[:self._fifo_index])
    history_next_observations = self._fifo_next_observations[self._fifo_index:]
    history_next_observations.extend(self._fifo_next_observations[:self._fifo_index])
    history_dones = self._fifo_dones[self._fifo_index:]
    history_dones.extend(self._fifo_dones[:self._fifo_index])
    return history_observations, history_actions, history_next_observations, history_dones

  def train(self, obs, actions, next_obs, dones):
    """Do one pass of training of the ICM."""

    # Split between train and validation data.
    __import__('ipdb').set_trace()
    n = len(obs)
    obs, actions, next_obs = np.array(obs), np.array(actions), np.array(next_obs)
    train_count = n // 2
    # If there's only one ensemble, then use all of the training data.
    if self._icm_model.ensemble_size < 2:
      train_count = n
    # train_count = (95 * n) // 100
    # obs_train, actions_train, next_obs_train = (
    #     obs[:train_count], actions[:train_count], next_obs[:train_count])
    # obs_valid, actions_valid, next_obs_valid = (
    #     obs[train_count:], actions[train_count:], next_obs[train_count:])

    # validation_data = ([np.array(obs_valid), np.array(actions_valid)],
    #                    np.array(next_obs_valid))

    # For each ensemble model, select a random subset of size n/2
    # (not necessarily contiguous) from the training sets and train on it.
    # This ensures that the ensemble models don't converge to the same
    # function over time.
    for i in range(self._icm_model.ensemble_size):
      rand_indices = np.random.permutation(obs.shape[0])[:train_count]
      obs_train, actions_train, next_obs_train = obs[rand_indices], actions[rand_indices], next_obs[rand_indices]
      self.fit(
          self._generate_batch(obs_train, actions_train, next_obs_train),
          i,
          steps_per_epoch=train_count // self._batch_size,
          epochs=self._num_epochs,
          validation_data=None)

    # Note: the same could possibly be achieved using parameters "callback",
    # "initial_epoch", "epochs" in fit_generator. However, this is not really
    # clear how this initial epoch is supposed to work.
    # TODO(damienv): change it to use callbacks of fit_generator.
    for _ in range(self._num_epochs):
      self._current_epoch += 1
      if self._checkpointer is not None:
        self._checkpointer.on_epoch_end(self._current_epoch)

  def _generate_batch(self, obs, actions, next_obs):
    """Generate batches of data used to train the ICM."""
    while True:
      # Train for one epoch.
      sample_count = obs.shape[0]
      number_of_batches = sample_count // self._batch_size
      for batch_index in range(number_of_batches):
        from_index = batch_index * self._batch_size
        to_index = (batch_index + 1) * self._batch_size
        yield (obs[from_index:to_index],
                actions[from_index:to_index],
                next_obs[from_index:to_index])

  def fit(self, gen, ensemble_index, steps_per_epoch, epochs, validation_data):
    for step in tqdm(range(steps_per_epoch * epochs)):
        obs, actions, next_obs = next(gen)
        forward_pred_error, inverse_pred_error = self._icm_model(ensemble_index, obs, actions, next_obs)
        loss = self._icm_model.loss_fn(forward_pred_error, inverse_pred_error)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
