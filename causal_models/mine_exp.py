"""Implements Mutual Information Neural Estimation."""
from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
from network_structure import interaction_v2


def get_t_network(state, actual_action, next_state, random_action):
  # state shape: [Batch, dims_s, embed]
  # action_input shape: [Batch, dims_a, embed]
  # t output shape: [Batch, dims_s, dims_a]
  x = keras.layers.concatenate([state, next_state])
  interaction = interaction_v2.Interaction(name='mine_t_interaction')

  raise NotImplementedError('support multiple actions.')
  t_actual = interaction([x, actual_action])
  t_random = interaction([x, random_action])
  return t_actual, t_random


def get_v_network(t_actual, t_random):
  v = keras.layers.subtract([
    keras.backend.mean(t_actual, axis=0),
    keras.backend.log(keras.backend.mean(keras.backend.exp(t_random), axis=0))], name='mine_v')
  return v
