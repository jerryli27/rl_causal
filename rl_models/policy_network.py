"""Provides an implementation of a general policy network"""

import gin
import gym
import numpy as np
from tensorflow import keras
from nn_utils import keras_utils


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


def get_network_for_action_embed(action_embed, name=''):
  output_dim = keras.backend.int_shape(action_embed)[1:]
  fc_dim = np.prod(output_dim)

  model = keras.models.Sequential([
    keras.layers.Dense(fc_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name='fc'),
    # keras.layers.Softmax(name='prob'),
    keras.layers.Reshape(output_dim),
  ], name=name)

  return model


@gin.configurable
def get_policy_network(state, allowed_actions, action_embed, num_hidden=2, hidden_dim=10):
  # Assumes state and goal_input has shape [batch, dim_s, embed_s]
  # assume action_shape = [batch, dim_a, embed_a]
  # s_g = keras.layers.concatenate([state, goal])
  model_layers = [
    keras.layers.Flatten(name='flatten'),
  ]
  for i in range(num_hidden):
    model_layers.append(
      keras.layers.Dense(hidden_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='elu', name='fc%d' %i))
  model = keras.models.Sequential(model_layers, name='policy_network_hidden')
  hidden_layer = model(state)
  # TODO: use allowed_actions
  mean_network = get_network_for_action_embed(action_embed, name='policy_mean')
  var_log_network = get_network_for_action_embed(action_embed, name='policy_var_log')
  mean = mean_network(hidden_layer)
  var_log = var_log_network(hidden_layer)
  var = keras.backend.exp(var_log)
  output = keras_utils.reparametrization(mean, var_log)

  output_dim = keras.backend.int_shape(action_embed)[1:]
  num_action_types = output_dim[0]
  allowed_actions_mask = keras.backend.expand_dims(keras.backend.expand_dims(
    keras.backend.sum(keras.backend.one_hot(allowed_actions, num_classes=num_action_types), axis=0), axis=0), axis=-1)
  output = keras.layers.Multiply(name='policy_network_masked_action_embed')([output, allowed_actions_mask])
  return mean, var, output


# @gin.configurable
# def get_policy_network(state, goal, allowed_actions, action_space, hidden_dim=10):
#   # Assumes state and goal_input has shape [batch, dim_s, embed_s]
#   # assume action_shape = [batch, dim_a, embed_a]
#   s_g = keras.layers.concatenate([state, goal])
#   model = keras.models.Sequential([
#     keras.layers.Flatten(name='flatten'),
#     keras.layers.Dense(hidden_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='elu', name='fc1'),
#   ], name='policy_network_hidden')
#   hidden_layer = model(s_g)
#   ret = {}
#   for i in allowed_actions:
#     output_network = get_network_for_action_space(action_space, i, name='policy_network_action_%d' %i)
#     output_prob = output_network(hidden_layer)
#     ret[i] = output_prob
#   return ret
