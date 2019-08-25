import collections
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from env_utils.spaces import me_dict_utils

GAMMA = 0.99

def _get_input_from_dict_space(space, name):
  keras_input = collections.OrderedDict()
  input_vec = collections.OrderedDict()

  for k, v in space.spaces.items():
    curr_name = '_'.join((name, k))
    keras_input[curr_name], input_vec[curr_name] = get_input_from_space(v, name=curr_name)
  return keras_input, input_vec

def get_input_from_space(space, name):
  """Given either an action_input space or a state space, creates and returns the network inputs."""

  if isinstance(space, gym.spaces.Dict):
    keras_input, input_vec = _get_input_from_dict_space(space, name)
  elif isinstance(space, me_dict_utils.MutuallyExclusiveDict):
    num_action_types = len(space.spaces)
    keras_input, input_vec = _get_input_from_dict_space(space, name)

    curr_name = name + '_type'
    assert curr_name not in space.spaces, '`%s` is reserved and cannot be used' %(curr_name)
    action_type_input = keras.layers.Input(shape=[], dtype='int32', name=curr_name)
    action_type_one_hot = tf.one_hot(action_type_input, num_action_types, name=curr_name + '_one_hot')
    keras_input[curr_name] = action_type_input
    input_vec[curr_name] = action_type_one_hot
  elif isinstance(space, gym.spaces.MultiDiscrete):
    shape = space.shape
    state_max_num_classes = np.max(space.nvec)
    keras_input = keras.layers.Input(shape=shape, dtype='int32', name=name)
    # input_one_hot = keras.backend.one_hot(keras_input, state_max_num_classes)
    input_vec = tf.one_hot(keras_input, state_max_num_classes, name=name+'_one_hot')
  else:
    raise NotImplementedError
  return keras_input, input_vec


def compute_discounted_cumulative_reward(rewards, gamma=GAMMA):
  assert rewards, 'empty rewards'
  ret = [0.0 for _ in range(len(rewards))]
  ret[-1] = rewards[-1]
  for i in range(len(rewards) - 2, -1, -1):
    ret[i] = ret[i + 1] * gamma + rewards[i]
  return ret
