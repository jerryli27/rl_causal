
# Some code are borrowed from https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py

import numpy as np
from tensorflow import keras
from env_utils import get_data_utils


LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
NOISE = 1.0  # Exploration noise
GAMMA = 0.99
ENTROPY_LOSS = 1e-3

def get_ppo_loss(policy, critic, reward_input, sampling_action_prob, actual_action_one_hot):
  """Return one single loss that does not depend on having a target."""
  advantage = reward_input - critic

  prob = policy * actual_action_one_hot
  old_prob = policy * sampling_action_prob
  r = prob / (old_prob + 1e-10)
  loss = -keras.backend.mean(keras.backend.minimum(r * advantage, keras.backend.clip(r, min_value=1 - LOSS_CLIPPING,
                                                                                       max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(
          prob * keras.backend.log(prob + 1e-10)))
  return loss


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


def get_action(policy_fn, x, action_space, is_eval=False):
  action_prob = policy_fn(x)[0]  # Batch size = 1
  # action_matrix = np.zeros(action_space)
  action = []

  for i in range(len(action_space.shape)):
    if is_eval is False:
      curr_action = np.random.choice(action_space[i], p=np.nan_to_num(action_prob[i]))
    else:
      curr_action = np.argmax(action_prob[i])
    # action_matrix[i, curr_action] = 1
    action.append(curr_action)
  action = np.array(action)
  return action, action, action_prob


def get_action_continuous(policy_fn, x, is_eval=False):
  action_prob = policy_fn(x)
  if is_eval is False:
    action = action_matrix = action_prob[0] + np.random.normal(loc=0, scale=NOISE, size=action_prob[0].shape)
  else:
    action = action_matrix = action_prob[0]
  return action, action_matrix, action_prob


def transform_reward(rewards, is_eval=False):
  # if is_eval is True:
  #   self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
  # else:
  #   self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
  for j in range(len(rewards) - 2, -1, -1):
    rewards[j] += rewards[j + 1] * GAMMA


def get_batch(policy_fn, env, num_data_points, is_continuous=False, is_eval=False):
  batch = [[], [], [], []]

  tmp_batch = [[], [], []]
  last_observation = get_data_utils.convert_env_observation(env.reset())
  rewards = []
  while len(batch[0]) < num_data_points:
    if is_continuous is False:
      action, action_matrix, predicted_action = get_action(policy_fn, get_data_utils.add_batch_dim(last_observation), env.action_space.nvec, is_eval=is_eval)  # TODO: use policy_fn etc.
    else:
      action, action_matrix, predicted_action = get_action_continuous(policy_fn, last_observation, is_eval=is_eval)
    observation, reward, done, info = env.step(action)
    observation = get_data_utils.convert_env_observation(observation)
    rewards.append(reward)

    tmp_batch[0].append(last_observation)
    tmp_batch[1].append(action_matrix)
    tmp_batch[2].append(predicted_action)
    last_observation = observation

    if done:
      transform_reward(rewards, is_eval=is_eval)
      if is_eval is False:
        for i in range(len(tmp_batch[0])):
          obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
          r = rewards[i]
          batch[0].append(obs)
          batch[1].append(action)
          batch[2].append(pred)
          batch[3].append(r)
      tmp_batch = [[], [], []]
      # reset_env
      last_observation = get_data_utils.convert_env_observation(env.reset())
      rewards = []

  obs, action, pred, reward = batch[0], np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]),
                                                                                                     (len(batch[3]), 1))
  # pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
  x = get_data_utils.combine_obs_dicts(obs)
  x.update({
    'action': action,
    'sampling_action_prob': pred,
    'reward': reward,
  })
  y = {
    'critic': reward,
    'tf_op_layer_Neg_1': keras.backend.zeros_like(reward),
  }
  return x, y
  # return obs, action, pred, reward