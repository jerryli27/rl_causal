
# Some code are borrowed from https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py

import numpy as np
from tensorflow import keras
from env_utils import env_rl_utils
from env_utils import get_data_utils
from nn_utils import policy_utils
from data_structures import option_utils
from network_structure import autoencoder_utils

LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
NOISE = 1.0  # Exploration noise
ENTROPY_LOSS = 1e-3

def get_ppo_loss(policy_mean, policy_var, critic, reward_input, sampling_mean, sampling_var, actual_action):
  # return get_ppo_loss_discrete(policy, critic, reward_input, allowed_actions, sampling_action_prob, actual_action_one_hot)
  return get_ppo_loss_continuous(policy_mean, policy_var, critic, reward_input, sampling_mean, sampling_var, actual_action)

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

def get_ppo_loss_continuous(policy_mean, policy_var, critic, reward_input, sampling_mean, sampling_var, actual_action):
  pi = keras.backend.constant(np.pi, dtype='float')
  advantage = reward_input - critic
  denom = keras.backend.sqrt(2 * pi * policy_var)
  prob_num = keras.backend.exp(- keras.backend.square(actual_action - policy_mean) / (2 * policy_var))
  old_prob_num = keras.backend.exp(- keras.backend.square(actual_action - sampling_mean) / (2 * sampling_var))

  prob = prob_num / denom
  old_prob = old_prob_num / denom
  r = prob / (old_prob + 1e-10)

  loss = -keras.backend.mean(keras.backend.minimum(r * advantage, keras.backend.clip(r, min_value=1 - LOSS_CLIPPING,
                                                                                     max_value=1 + LOSS_CLIPPING) * advantage))

  loss = keras.layers.Lambda(lambda x: x, name='ppo_loss')(loss)  # Hacky way to give it a name.
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
    NOISE = 1.0  # Exploration noise
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


def get_ppo_action_fn(policy_fn, is_eval=False):
  def get_action_fn(obs):
    converted = get_data_utils.convert_env_observation(obs, add_batch_dim=True)
    policy_outputs = policy_fn(converted)
    policy_outputs = get_data_utils.remove_batch_dim(policy_outputs)
    policy_mean, policy_var, policy_action = policy_outputs[:3]
    policy_action_vec_decoded = np.array(policy_outputs[3:])
    action_type = autoencoder_utils.get_action_type_from_action_embed(policy_action)

    action_prob = policy_action_vec_decoded[action_type]
    # TODO: do different things based on discrete vs continuous.
    if is_eval:
      action_val = np.argmax(action_prob, axis=-1)
    else:
      num_possible_actions = action_prob.shape[-1]
      action_val = []
      for sub_action_type in range(action_prob.shape[0]):
        action_val.append(np.random.choice(num_possible_actions, p=np.nan_to_num(action_prob[sub_action_type])))
      action_val = np.array(action_val)
    action = (action_type, action_val)
    action_info = policy_mean, policy_var, policy_action
    return action, action_info
  return get_action_fn


def get_ppo_data(policy_fn, state_input, action_input, env, num_episodes, gamma, is_eval=False, render=False):
  # TODO(jryli): Scale the reward to between 0 and 1, and constrain the critic to be in between the two as well.
  # TODO(jryli): there is a discrepancy between training and the trained option network's goal. Maybe have a network that sets harder goals and collect data through that?

  # For each observation, feed it in to policy_fn() to get the action prob and actual actions.
  # Perform the action (what if the action is an option? env should take care of it) and record the observed results.
  x, _ = get_data_utils.get_data(num_episodes, state_input, action_input, gamma, env=env, get_action_fn=get_ppo_action_fn(policy_fn, is_eval=is_eval), add_random_actions=False, render=render)
  x_augmented, _ = get_data_utils.get_data(num_episodes, state_input, action_input, gamma, env=env, get_action_fn=get_ppo_action_fn(policy_fn, is_eval=is_eval), add_random_actions=False, use_augmented_goal=True, render=render)
  for k in x_augmented.keys():
    x[k] = np.concatenate((x[k], (x_augmented[k])), axis=0)
  y = {
    'critic_loss': keras.backend.zeros_like(x['state_reward']),
    'ppo_loss': keras.backend.zeros_like(x['state_reward']),
  }
  return x, y

  # trajectories = {
  #   'state': [],
  #   'goal': [],
  #   'action': [],
  #   'action_prob': [],
  #   'reward': [],
  # }
  #
  # last_observation = get_data_utils.convert_env_observation(env.reset())
  # option = option_utils.Option(
  #   policy_fn=policy_fn,
  #   env=env,
  #   allowed_action_types=allowed_action_types,
  #   termination_fn=lambda _: False,
  # )
  # while len(trajectories['reward']) < num_data_points:
  #   # TODO:
  #   raise NotImplementedError('generate goal.')
  #   trajectory = option.run_until_termination(last_observation, is_eval=is_eval)
  #   for k in trajectories.keys():
  #     trajectories[k].extend(trajectory[k])
  #   # reset_env
  #   last_observation = get_data_utils.convert_env_observation(env.reset())
  #
  # trajectories['reward'] = np.reshape(trajectories['reward'], (len(trajectories['reward']), 1))
  # x = get_data_utils.combine_obs_dicts(trajectories['state'])
  # x.update({
  #   'action': np.array(trajectories['action']),
  #   'sampling_action_prob': np.array((trajectories['action_prob'])),
  #   'reward': trajectories['reward'],
  # })
  # y = {
  #   'critic': trajectories['reward'],
  #   'ppo_loss': keras.backend.zeros_like(trajectories['reward']),
  # }
  # return x, y