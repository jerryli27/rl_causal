"""A model that predicts the intervention distribution of performing an action"""
import numpy as np
import scipy.stats
from tensorflow import keras

class InterventionModel(object):
  """Given (s,a) and a predictive model for (s_intervened, s'), reason about s_intervened.

  In RL, the s_intervened will not be shown directly. Instead we observe the next state s'. Thus this module is not
  meant to be trained directly, but in combination with a predictive model.
  """

  def __init__(self, state_shape, action_shape, model_type):
    """Initialize the model."""
    self.input_state_shape = state_shape
    self.output_state_shape = state_shape  # Assumes to be the same.
    self.action_shape = action_shape
    self.model_type = model_type

    self._prepare_model()

  def _prepare_model(self):
    raise NotImplementedError('not quite there yet. It needs to take the action as well and predict the intervention density for each action.')
    # Assume for now that there is no covariance matrix, and the intervention is deterministic.
    if self.model_type == 'fc':
      assert len(self.input_shape) == 1, 'wrong input shape'
      assert len(self.output_shape) == 1, 'wrong output shape'
      # Hard coded dimensions for now.
      self.model = keras.models.Sequential([
        keras.layers.Dense(32, input_shape=self.input_shape),
        keras.layers.Activation('relu'),
        keras.layers.Dense(self.output_shape[0]),
        keras.layers.Activation('softmax'),
      ])
      # self.model.compile(
      #   optimizer='adagrad',
      #   loss='binary_crossentropy',
      #   metrics=['accuracy'])
    else:
      raise NotImplementedError('Model type %s is not supported' % self.model_type)

  def train(self, input_tensor, expected_output_tensor):
    raise NotImplementedError('This model is not meant to be trained on its own.')


  def get_model(self):
    return self.model


def _get_action_embed(one_hot_action, num_possible_actions, output_dim, name_postfix, embeddings_regularizer=None):
  action_embed_lookup = keras.layers.Embedding(
    input_dim=num_possible_actions, output_dim=output_dim,
    name='action_embed'+name_postfix, embeddings_regularizer=embeddings_regularizer)
  action_embed = action_embed_lookup(one_hot_action)
  # action_embed = keras.backend.squeeze(action_embed, axis=1)
  return action_embed


def get_intervention_model_one_hot(input_state_logits, input_action, num_possible_actions):
  # Assumes input_state is one-hot with shape (batch, dims, one_hot_state)
  state_shape = keras.backend.int_shape(input_state_logits)
  # Assumes actions are one-hot encoded (batch, one_hot_action) and each action has an embedding.
  action_shape = keras.backend.int_shape(input_action)
  action_emb_output_dim = np.prod(state_shape[1:])

  # TODO: for general non-one-hot actions, perhaps using the fully connected layer
  # is better...
  action_emb_weight = _get_action_embed(input_action, num_possible_actions, output_dim=action_emb_output_dim, embeddings_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name_postfix='_weight')
  action_emb_bias = _get_action_embed(input_action, num_possible_actions,  output_dim=action_emb_output_dim, embeddings_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name_postfix='_bias')

  action_embed_reshape_op = keras.layers.Reshape(state_shape[1:])
  action_emb_weight = action_embed_reshape_op(action_emb_weight)
  action_emb_bias = action_embed_reshape_op(action_emb_bias)
  #

  # wx+b model
  # intervened_state_logits = keras.layers.multiply([input_state_logits, action_emb_weight],
  #                                                 name='intervened_state_logits_no_bias')
  # intervened_state_logits = keras.layers.add(
  #   [input_state_logits, intervened_state_logits, action_emb_bias],
  #   name='intervened_state_logits')

  # +b model
  # intervened_state_logits = keras.layers.add(
  #   [input_state_logits, action_emb_bias],
  #   name='intervened_state_logits')

  # Identity model
  intervened_state_logits = input_state_logits

  # intervened_state_logits = keras.layers.multiply([input_state_logits, action_emb_weight],
  #                                                 name='intervened_state_logits_residule')
  # intervened_state_logits = keras.layers.add(
  #   [input_state_logits, intervened_state_logits],
  #   name='intervened_state_logits')

  # Note that the input state is a probability distribution. We need the output to maintain that property.

  return intervened_state_logits


def infer_action(intervened_state_one_hot, input_state, input_state_one_hot, input_action, data, p_value_threshold=0.05):
  """Given a trained model, output all the causal edges between each pair of (action, state)."""
  # T
  diff_layer = keras.layers.subtract([input_state_one_hot, intervened_state_one_hot])
  # diff = keras.backend.abs(diff)
  # model = keras.models.Model(inputs=[input_state, input_action], outputs=[diff_layer])
  # model.compile(optimizer='adam')
  # diff = model.predict(data)
  predict = keras.backend.function([input_state, input_action], diff_layer)
  diff = predict([data['current_state'], data['action']])

  # Assume the diffs follow a gaussian distribution. we say that the action causes the
  # state change if 0 is out of the 95% confidence interval of the mean.
  t_test_result = scipy.stats.ttest_1samp(diff, 0)
  # TODO: use p_value_threshold
  return t_test_result


