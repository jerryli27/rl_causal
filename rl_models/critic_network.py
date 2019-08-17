"""Provides an implementation of a general policy network"""

import numpy as np
from tensorflow import keras

def get_critic_network(state, goal):
  # Assumes state and goal has shape [batch, dim_s, embed_s]
  # assume action_shape = [batch, dim_a, embed_a]
  hidden_dim = 100
  output_dim = 1

  s_g = keras.layers.concatenate([state, goal])
  critic = keras.models.Sequential([
    keras.layers.Flatten(name='flatten'),
    keras.layers.Dense(hidden_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='elu', name='fc1'),
    keras.layers.Dense(output_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='elu', name='fc1'),
  ], name='critic')
  v = critic(s_g)
  return v