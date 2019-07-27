"""A model that predicts the intervention distribution of performing an action"""
import tensorflow as tf
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

def ge_intervention_model(input_state, input_action, num_possible_actions):
  state_shape = keras.backend.int_shape(input_state)
  action_shape = keras.backend.int_shape(input_action)
  assert action_shape[1] == 1

  # Assumes actions are one-hot encoded and each action has an embedding.
  action_embed_lookup = keras.layers.Embedding(input_dim=num_possible_actions, output_dim=state_shape[1], input_length=action_shape[1], name='action_embed')
  action_embed = action_embed_lookup(input_action)
  action_embed = keras.backend.squeeze(action_embed, axis=1)

  # embedding_model = keras.models.Model(inputs=input_action, outputs=action_embed)
  intervened_state = keras.layers.add([input_state, action_embed], name='intervened_state')

  # if model_type == 'fc':
  #   assert len(input_shape) == 1, 'wrong input shape'
  #   assert len(output_shape) == 1, 'wrong output shape'
  #   # Hard coded dimensions for now.
  #   model = keras.models.Sequential([
  #     keras.layers.Dense(32),
  #     keras.layers.Activation('relu'),
  #     keras.layers.Dense(output_shape[0]),
  #     keras.layers.Activation('softmax'),
  #   ])
  #   # model.compile(
  #   #   optimizer='adagrad',
  #   #   loss='binary_crossentropy',
  #   #   metrics=['accuracy'])
  # else:
  #   raise NotImplementedError('Model type %s is not supported' % model_type)

  # model = keras.models.Model(inputs=[input_state, input_action], outputs=[intervened_state])
  # return model
  return intervened_state