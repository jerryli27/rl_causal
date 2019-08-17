"""Provides an implementation of a general policy network"""

import numpy as np
from tensorflow import keras

def get_policy_network(state, goal, action_shape):
  # Assumes state and goal has shape [batch, dim_s, embed_s]
  # assume action_shape = [batch, dim_a, embed_a]
  s_g = keras.layers.concatenate([state, goal])
  output_dim = np.prod(action_shape)
  model = keras.models.Sequential([
    keras.layers.Flatten(name='flatten'),
    keras.layers.Dense(output_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='elu', name='fc1'),
    keras.layers.Reshape(action_shape, name='logits'),
    keras.layers.Softmax(name='prob'),
  ], name='policy_network')
  output_action = model(s_g)
  return output_action