import gym
import numpy as np

from env_utils import get_data_utils


def get_one_action(action_space, action_type_i, action_prob, is_eval):
  if isinstance(action_space, gym.spaces.MultiDiscrete):
    num_possible_actions = action_space.nvec[action_type_i]
    if is_eval is False:
      curr_action = np.random.choice(num_possible_actions, p=np.nan_to_num(action_prob))
    else:
      curr_action = np.argmax(action_prob)
  else:
    raise NotImplementedError
  return curr_action


def get_one_random_action_and_prob(action_space, action_type_i):
  if isinstance(action_space, gym.spaces.MultiDiscrete):
    num_possible_actions = action_space.nvec[action_type_i]
    action = np.random.choice(num_possible_actions)
    action_prob = np.ones((num_possible_actions,), dtype=np.float32) / num_possible_actions
  else:
    raise NotImplementedError
  return action, action_prob


def get_actions(policy_fn, x, action_space, allowed_action_types, is_eval=False):
  # TODO: add continuous support.
  policy_action_prob = get_data_utils.remove_batch_dim(policy_fn(x))  # Batch size = 1
  actions = []
  action_probs = []
  for action_type_i in range(len(action_space.nvec)):
    if action_type_i in allowed_action_types:
      curr_action = get_one_action(action_space, action_type_i, policy_action_prob[action_type_i], is_eval)
      curr_action_prob = policy_action_prob[action_type_i]
    else:
      curr_action, curr_action_prob = get_one_random_action_and_prob(action_space, action_type_i)
    actions.append(curr_action)
    action_probs.append(curr_action_prob)
  actions = np.array(actions)
  policy_action_prob = np.array(action_probs)
  return actions, actions, policy_action_prob


def get_actions_continuous(policy_fn, x, is_eval=False):
  NOISE = 1.0  # Exploration noise
  action_prob = policy_fn(x)
  if is_eval is False:
    action = action_matrix = action_prob[0] + np.random.normal(loc=0, scale=NOISE, size=action_prob[0].shape)
  else:
    action = action_matrix = action_prob[0]
  return action, action_matrix, action_prob