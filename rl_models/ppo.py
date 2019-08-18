
# Some code are borrowed from https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py

import numpy as np
from tensorflow import keras
from env_utils import env_rl_utils
from env_utils import get_data_utils
from nn_utils import policy_utils

LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
NOISE = 1.0  # Exploration noise
ENTROPY_LOSS = 1e-3

def get_ppo_loss(policy, critic, reward_input, allowed_actions, sampling_action_prob, actual_action_one_hot):
  return get_ppo_loss_discrete(policy, critic, reward_input, allowed_actions, sampling_action_prob, actual_action_one_hot)

def get_ppo_loss_discrete(policy, critic, reward_input, allowed_actions, sampling_action_prob, actual_action_one_hot, epsilon=1e-10):
  """Return one single loss that does not depend on having a target."""
  advantage = reward_input - critic

  prob = 1.0
  old_prob = 1.0
  for i in allowed_actions:
    prob *= policy[i] * actual_action_one_hot[:, i]
    old_prob *= policy[i] * sampling_action_prob[:, i]
  r = prob / (old_prob + epsilon)
  loss = -keras.backend.mean(keras.backend.minimum(r * advantage, keras.backend.clip(r, min_value=1 - LOSS_CLIPPING,
                                                                                       max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(
          prob * keras.backend.log(prob + epsilon)))
  loss = keras.layers.Lambda(lambda x: x, name='ppo_loss')(loss)  # Hacky way to give it a name.
  return loss

def get_ppo_loss_continuous():
  raise NotImplementedError


def proximal_policy_optimization_loss(advantage, old_prediction):
  def loss(y_true, y_pred):
    prob = y_true * y_pred
    old_prob = y_true * old_prediction
    r = prob / (old_prob + 1e-10)
    return -keras.backend.mean(keras.backend.minimum(r * advantage, keras.backend.clip(r, min_value=1 - LOSS_CLIPPING,
                                                                                       max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(
          prob * keras.backend.log(prob + 1e-10)))

  return loss


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
  def loss(y_true, y_pred):
    var = keras.backend.square(NOISE)
    pi = 3.1415926
    denom = keras.backend.sqrt(2 * pi * var)
    prob_num = keras.backend.exp(- keras.backend.square(y_true - y_pred) / (2 * var))
    old_prob_num = keras.backend.exp(- keras.backend.square(y_true - old_prediction) / (2 * var))

    prob = prob_num / denom
    old_prob = old_prob_num / denom
    r = prob / (old_prob + 1e-10)

    return -keras.backend.mean(keras.backend.minimum(r * advantage, keras.backend.clip(r, min_value=1 - LOSS_CLIPPING,
                                                                                       max_value=1 + LOSS_CLIPPING) * advantage))

  return loss


def get_batch(policy_fn, env, num_data_points, allowed_action_types, is_continuous=False, is_eval=False):
  # TODO(jryli): Add goal modification to make the task a bit easier.
  # TODO(jryli): Scale the reward to between 0 and 1, and constrain the critic to be in between the two as well.
  # TODO(jryli): there is a discrepancy between training and the trained option network's goal. Maybe have a network that sets harder goals and collect data through that?
  # TODO(jryli): allow default no-op actions.
  batch = [[], [], [], []]

  tmp_batch = [[], [], []]
  last_observation = get_data_utils.convert_env_observation(env.reset())
  rewards = []
  while len(batch[0]) < num_data_points:
    if is_continuous is False:
      action, action_matrix, action_prob = policy_utils.get_actions(policy_fn, get_data_utils.add_batch_dim(last_observation), env.action_space, allowed_action_types, is_eval=is_eval)
    else:
      action, action_matrix, action_prob = policy_utils.get_actions_continuous(policy_fn, last_observation, is_eval=is_eval)
    observation, reward, done, info = env.step(action)
    observation = get_data_utils.convert_env_observation(observation)
    rewards.append(reward)

    tmp_batch[0].append(last_observation)
    tmp_batch[1].append(action_matrix)
    tmp_batch[2].append(action_prob)
    last_observation = observation

    if done:
      rewards = env_rl_utils.compute_discounted_cumulative_reward(rewards)
      if is_eval is False:
        for i in range(len(tmp_batch[0])):
          obs, action, action_prob = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
          r = rewards[i]
          batch[0].append(obs)
          batch[1].append(action)
          batch[2].append(action_prob)
          batch[3].append(r)
      tmp_batch = [[], [], []]
      # reset_env
      last_observation = get_data_utils.convert_env_observation(env.reset())
      rewards = []

  # TODO(jryli): maybe truncate everything to be len == num_data_points
  obs, action, action_prob, reward = batch[0], np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]),
                                                                                                     (len(batch[3]), 1))
  # pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
  x = get_data_utils.combine_obs_dicts(obs)
  x.update({
    'action': action,
    'sampling_action_prob': action_prob,
    'reward': reward,
  })
  y = {
    'critic': reward,
    'ppo_loss': keras.backend.zeros_like(reward),
  }
  return x, y