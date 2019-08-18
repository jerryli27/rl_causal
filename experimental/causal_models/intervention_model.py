"""A model that predicts the intervention distribution of performing an action_input.

It assumes that the action_input is represented by [batch, types_dim, embed_dim] where each type of action_input is independent of
the other types and actions within the same type share some commonalities. 
"""
import numpy as np
import scipy.stats
from tensorflow import keras

def _get_action_embed(one_hot_action, num_possible_actions, output_dim, name_postfix, embeddings_regularizer=None):
  action_embed_lookup = keras.layers.Embedding(
    input_dim=num_possible_actions, output_dim=output_dim,
    name='action_embed'+name_postfix, embeddings_regularizer=embeddings_regularizer)
  action_embed = action_embed_lookup(one_hot_action)
  # action_embed = keras.backend.squeeze(action_embed, axis=1)
  return action_embed

# def get_action_embed_one_hot(input_action, num_possible_actions):


def get_intervention_model_one_hot(input_state, input_action, num_possible_actions):
  # Assumes input_state is one-hot with shape (batch, dims, one_hot_state)
  state_shape = keras.backend.int_shape(input_state)
  # Assumes actions are one-hot encoded (batch, one_hot_action) and each action_input has an embedding.
  action_shape = keras.backend.int_shape(input_action)
  assert action_shape[1] == 1  # TODO: for multiple action_input dims, I can add their interventions together...
  action_emb_output_dim = np.prod(state_shape[1:])

  # TODO: for general non-one-hot actions, perhaps using the fully connected layer
  # is better...
  action_emb_weight = _get_action_embed(input_action, num_possible_actions, output_dim=action_emb_output_dim, embeddings_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name_postfix='_weight')
  action_emb_bias = _get_action_embed(input_action, num_possible_actions,  output_dim=action_emb_output_dim, embeddings_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name_postfix='_bias')

  action_embed_reshape_op = keras.layers.Reshape(state_shape[1:])
  action_emb_weight = action_embed_reshape_op(action_emb_weight)
  action_emb_bias = action_embed_reshape_op(action_emb_bias)
  #

  # res(wx+b) model
  intervened_state = keras.layers.multiply([input_state, action_emb_weight],
                                           name='intervened_state_logits_no_bias')
  intervened_state = keras.layers.add(
    [input_state, intervened_state, action_emb_bias],
    name='intervened_state')


  # Identity model
  # intervened_state = input_state

  # intervened_state = keras.layers.multiply([input_state_logits, action_emb_weight],
  #                                                 name='intervened_state_logits_residule')
  # intervened_state = keras.layers.add(
  #   [input_state_logits, intervened_state],
  #   name='intervened_state')

  # Note that the input state is a probability distribution. We need the output to maintain that property.

  return intervened_state


def infer_action(intervened_state_one_hot, input_state, input_state_one_hot, input_action, data_with_action, data_with_random_action, p_value_threshold=0.05):
  """Given a trained model, output all the causal edges between each pair of (action_input, state)."""
  # T
  diff_layer = keras.layers.subtract([input_state_one_hot, intervened_state_one_hot])
  # diff = keras.backend.abs(diff)
  # model = keras.models.Model(inputs=[input_state, input_action], outputs=[diff_layer])
  # model.compile(optimizer='adam')
  # diff = model.predict(data)
  predict = keras.backend.function([input_state, input_action], diff_layer)
  test_diff = predict([data_with_action['state_input'], data_with_action['action_input']])
  base_diff = predict([data_with_random_action['state_input'], data_with_random_action['action_input']])

  # Assume the diffs follow a gaussian distribution. we say that the action_input causes the
  # state change if 0 is out of the 95% confidence interval of the mean.
  t_test_result = scipy.stats.ttest_ind(base_diff, test_diff, equal_var=False)
  # TODO: use p_value_threshold
  return t_test_result


