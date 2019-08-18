"""A model that predicts the intervention distribution of performing an action_input.

It assumes that the action_input is represented by [batch, types_dim, embed_dim] where each type of action_input is independent of
the other types and actions within the same type share some commonalities. States are represented in a similar fashion.
It further assumes that each action_input acts as a function to change the state before the state got transported into the
next state through the transition function: s->interfered_state->s'.
There may be a fixed embedding that represents the "NULL" action_input where the agent does nothing.
"""
import gin
import numpy as np
import tensorflow as tf
from tensorflow import keras


class InterventionModel(keras.layers.Layer):
  def __init__(self, hidden_dim=10, **kwargs):
    # super(keras.layers.Layer, self).__init__(**kwargs)
    super().__init__(**kwargs)
    self.hidden_dim = hidden_dim
    self.model_array = None
    self.no_state_diff = None

  def build(self, input_shape):
    # Inputs are: (state_embed, action_embed, causal_relation)
    unused_batch, self.s_types_dim, self.s_embed_dim = input_shape[0]
    _, self.a_types_dim, self.a_embed_dim = input_shape[1]
    assert self.s_types_dim, self.a_types_dim == input_shape[2]
    if self.model_array is None:
      self.model_array = []
      for i in range(self.s_types_dim):
        s_models = []
        for j in range(self.a_types_dim):
          curr_type_model = keras.models.Sequential([
            keras.layers.Dense(self.hidden_dim, activation=keras.activations.relu,
                               kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name='fc1'),
            keras.layers.Dense(self.hidden_dim, activation=keras.activations.relu,
                               kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name='fc2'),
            keras.layers.Dense(self.s_embed_dim, activation=keras.activations.relu,
                               kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name='fc3'),
          ], name='intervention_model_%d_%d' % (i, j))
          s_models.append(curr_type_model)
        self.model_array.append(s_models)

    # if self.no_state_diff is None:
    #   self.no_state_diff = keras.backend.zeros((1, self.s_embed_dim), name='no_state_diff')

  def call(self, inputs, **kwargs):
    """
    :param inputs: (state_embed, action_embed, causal_relation)
      state_embed: [batch, s_types_dim, s_embed_dim]
      action_embed: [batch, a_types_dim, a_embed_dim]
      causal_relation: Boolean of shape [s_types_dim, a_types_dim]
    :return: next_state_embed [batch, s_types_dim, s_embed_dim]
    """

    state_embed, action_embed, causal_relation = inputs
    batch_size = keras.backend.int_shape(state_embed)[0]
      # self.no_state_diff = keras.backend.zeros((batch, s_embed_dim))  # Doesn't work for some reason. probably reinit vars?
      # self.no_state_diff = np.zeros((batch, s_embed_dim), dtype=np.float32)
    # state_diffs = []
    state_diffs = tf.TensorArray(tf.float32, size=self.s_types_dim)
    for i in range(self.s_types_dim):
      # sliced_state_diffs = []
      sliced_state_diffs = tf.TensorArray(tf.float32, size=self.a_types_dim)
      for j in range(self.a_types_dim):
        if causal_relation[i, j]:
          curr_type_model = self.model_array[i][j]
          curr_action_embed = keras.backend.reshape(action_embed[:, j], [-1, self.a_embed_dim])
          sliced_state_diff = curr_type_model(curr_action_embed)
          # sliced_state_diff = curr_type_model(action_embed[:, j])
          # sliced_state_diffs.append(sliced_state_diff)
        else:
          sliced_state_diff = keras.backend.zeros_like(state_embed)[:,0,:]
          # sliced_state_diff = keras.backend.repeat_elements(self.no_state_diff, rep=batch_size, axis=0)
          # sliced_state_diff = self.no_state_diff
        sliced_state_diffs = sliced_state_diffs.write(j, sliced_state_diff)

      # if len(sliced_state_diffs) == 1:
      #   sliced_state_diffs_sum = sliced_state_diffs[0]
      # elif len(sliced_state_diffs) > 1:
      #   sliced_state_diffs_sum = keras.layers.add(sliced_state_diffs)
      # else:
      #   sliced_state_diffs_sum = keras.backend.zeros((batch, s_embed_dim))
      # state_diffs.append(sliced_state_diffs_sum)

      sliced_state_diffs_sum = sliced_state_diffs.stack()  # Stacked on the first dim
      sliced_state_diffs_sum = keras.backend.sum(sliced_state_diffs_sum, axis=0)
      state_diffs.write(i, sliced_state_diffs_sum)

    # state_diffs = keras.backend.stack(state_diffs, axis=1)
    state_diffs = state_diffs.stack()  # Stacked on the first dim
    state_diffs = keras.backend.permute_dimensions(state_diffs, pattern=(1, 0, 2))
    next_state_embed = keras.layers.add([state_embed, state_diffs])
    return next_state_embed





# The following implementation also works, but was pretty fragile due to tf.function. I got error
# An op outside of the function building code is being passed a "Graph" tensor.
# class InterventionModel(object):
#   def __init__(self):
#     self.model_array = None
#     self.no_state_diff = None
#
#   @tf.function
#   @gin.configurable
#   def __call__(self, state_embed, action_embed, causal_relation, hidden_dim=10):
#     """
#
#     :param state_embed: [batch, s_types_dim, s_embed_dim]
#     :param action_embed: [batch, a_types_dim, a_embed_dim]
#     :param causal_relation: Boolean of shape [s_types_dim, a_types_dim]
#     :return: next_state_embed [batch, s_types_dim, s_embed_dim]
#     """
#     batch, s_types_dim, s_embed_dim = keras.backend.int_shape(state_embed)
#     _, a_types_dim, a_embed_dim = keras.backend.int_shape(action_embed)
#     assert s_types_dim, a_types_dim == keras.backend.int_shape(causal_relation)
#     if self.model_array is None:
#       self.model_array = []
#       for i in range(s_types_dim):
#         s_models = []
#         for j in range(a_types_dim):
#           curr_type_model = keras.models.Sequential([
#             keras.layers.Dense(hidden_dim, activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name='fc1'),
#             keras.layers.Dense(hidden_dim, activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name='fc2'),
#             keras.layers.Dense(s_embed_dim, activation=keras.activations.relu, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name='fc3'),
#           ], name='intervention_model_%d_%d' %(i, j))
#           s_models.append(curr_type_model)
#         self.model_array.append(s_models)
#
#     if self.no_state_diff is None:
#       self.no_state_diff = keras.backend.zeros_like(state_embed)[:, 0, :]
#       # self.no_state_diff = keras.backend.zeros((batch, s_embed_dim))  # Doesn't work for some reason. probably reinit vars?
#       # self.no_state_diff = np.zeros((batch, s_embed_dim), dtype=np.float32)
#     # state_diffs = []
#     state_diffs = tf.TensorArray(tf.float32, size=s_types_dim)
#     for i in range(s_types_dim):
#       # sliced_state_diffs = []
#       sliced_state_diffs = tf.TensorArray(tf.float32, size=a_types_dim)
#       for j in range(a_types_dim):
#         if causal_relation[i, j]:
#           curr_type_model = self.model_array[i][j]
#           sliced_state_diff = curr_type_model(action_embed[:, j])
#           # sliced_state_diffs.append(sliced_state_diff)
#         else:
#           sliced_state_diff = self.no_state_diff
#         sliced_state_diffs = sliced_state_diffs.write(j, sliced_state_diff)
#
#       # if len(sliced_state_diffs) == 1:
#       #   sliced_state_diffs_sum = sliced_state_diffs[0]
#       # elif len(sliced_state_diffs) > 1:
#       #   sliced_state_diffs_sum = keras.layers.add(sliced_state_diffs)
#       # else:
#       #   sliced_state_diffs_sum = keras.backend.zeros((batch, s_embed_dim))
#       # state_diffs.append(sliced_state_diffs_sum)
#
#       sliced_state_diffs_sum = sliced_state_diffs.stack()  # Stacked on the first dim
#       sliced_state_diffs_sum = keras.backend.sum(sliced_state_diffs_sum, axis=0)
#       state_diffs.write(i, sliced_state_diffs_sum)
#
#     # state_diffs = keras.backend.stack(state_diffs, axis=1)
#     state_diffs = state_diffs.stack()  # Stacked on the first dim
#     state_diffs = keras.backend.permute_dimensions(state_diffs, pattern=(1, 0, 2))
#     next_state_embed = keras.layers.add([state_embed, state_diffs])
#     return next_state_embed
