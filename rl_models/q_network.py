"""Provides an implementation of a general policy network"""

import numpy as np
from tensorflow import keras

def get_q_network(state, goal, action):
  # Assumes state and goal has shape [batch, dim_s, embed_s]
  # assume action_shape = [batch, dim_a, embed_a]
  hidden_dim = 100

  s_g = keras.layers.concatenate()([state, goal])
  state_encoder = keras.models.Sequential([
    keras.layers.Flatten(name='flatten'),
    keras.layers.Dense(hidden_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='elu', name='fc1'),
  ], name='q_network_state_encoder')
  action_encoder = keras.models.Sequential([
    keras.layers.Flatten(name='flatten'),
    keras.layers.Dense(hidden_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='elu', name='fc1'),
  ], name='q_network_action_encoder')
  state_embedding = state_encoder(s_g)
  action_embedding = action_encoder(action)
  output_q = keras.backend.batch_dot(state_embedding, action_embedding, axis=1)
  return output_q