import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from env_utils.spaces import me_dict_utils

GAMMA = 0.99

def get_input_from_space(space, name):
  """Given either an action_input space or a state space, creates and returns the network inputs."""
  if isinstance(space, me_dict_utils.MutuallyExclusiveDict):
    num_action_types = len(space.spaces)

    assert 'action_type' not in space.spaces, '`action_type` is reserved and cannot be used'
    action_type_input = keras.layers.Input(shape=[1], dtype='int32', name=name + '_action_type')
    action_type_one_hot = tf.one_hot(action_type_input, num_action_types, name=name + '_action_type_one_hot')
    keras_input = {'action_type': action_type_input}
    input_vec = {'action_type': action_type_one_hot}

    for k, v in space.spaces.items():
      keras_input[k], input_vec[k] = get_input_from_space(v, name='_'.join((name, k)))
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
