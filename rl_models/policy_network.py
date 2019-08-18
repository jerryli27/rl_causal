"""Provides an implementation of a general policy network"""

import gin
import gym
import numpy as np
from tensorflow import keras


def get_network_for_action_space(action_space, index, name=''):
  if isinstance(action_space, gym.spaces.MultiDiscrete):
    output_dim = action_space.nvec[index]
    model = keras.models.Sequential([
      keras.layers.Dense(output_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name='logits'),
      keras.layers.Softmax(name='prob'),
    ], name=name)
    return model
  else:
      raise NotImplementedError


@gin.configurable
def get_policy_network(state, goal, allowed_actions, action_space, hidden_dim=10):
  # Assumes state and goal_input has shape [batch, dim_s, embed_s]
  # assume action_shape = [batch, dim_a, embed_a]
  s_g = keras.layers.concatenate([state, goal])
  model = keras.models.Sequential([
    keras.layers.Flatten(name='flatten'),
    keras.layers.Dense(hidden_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='elu', name='fc1'),
  ], name='policy_network_hidden')
  hidden_layer = model(s_g)
  ret = {}
  for i in allowed_actions:
    output_network = get_network_for_action_space(action_space, i, name='policy_network_action_%d' %i)
    output_prob = output_network(hidden_layer)
    ret[i] = output_prob
  return ret




# def get_policy_network(state, goal, allowed_actions, action_spaces, hidden_dim=10):
#   # Assumes state and goal_input has shape [batch, dim_s, embed_s]
#   # assume action_shape = [batch, dim_a, embed_a]
#   s_g = keras.layers.concatenate([state, goal])
#   model = keras.models.Sequential([
#     keras.layers.Flatten(name='flatten'),
#     keras.layers.Dense(hidden_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='elu', name='fc1'),
#     keras.layers.Reshape(action_shape, name='logits'),
#     keras.layers.Softmax(name='prob'),
#   ], name='policy_network')
#   output_action = model(s_g)
#   return output_action