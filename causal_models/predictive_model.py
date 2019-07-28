"""A predictive model with causal reasoning abilities under the RL framework."""
import numpy as np
from tensorflow import keras


class PredictiveModel(object):
  """Given (s,s'), learns the causal relationship between the two.

  In an RL framework, s will be the output of the intervention model -- the s_intervened which comes from f(s_obs, a).
  """
  def __init__(self, input_shape, output_shape, model_type, intervention_model):
    """Initialize the model."""
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.model_type = model_type
    # TODO(jryli): Clean up. In keras there seems to be a clear separation between model and training.
    self.intervention_model = intervention_model

    self.model_input = keras.layers.Input(shape=self.input_shape)
    self.intervention_output = self.intervention_model.model(self.model_input)

    self._prepare_model()


  def _prepare_model(self):
    if self.model_type == 'fc':
      # Hard coded dimensions for now.

      # Assume for now that there is no covariance matrix
      if self.model_type == 'fc':
        assert len(self.input_shape) == 1, 'wrong input shape'
        assert len(self.output_shape) == 1, 'wrong output shape'
        # Hard coded dimensions for now.
        self.prediction_model = keras.models.Sequential([
          keras.layers.Dense(32, input_shape=self.input_shape),
          keras.layers.Activation('relu'),
          keras.layers.Dense(self.output_shape[0]),
          keras.layers.Activation('softmax'),
        ])
        # self.prediction_model.compile(
        #   optimizer='adagrad',
        #   loss='binary_crossentropy',
        #   metrics=['accuracy'])
    else:
      raise NotImplementedError('Model type %s is not supported' %self.model_type)

    self.prediction = self.prediction_model(self.intervention_output)
    self.model = keras.models.Model(inputs=[self.model_input], outputs=[self.prediction])
    self.model.compile(
          optimizer='adagrad',
          loss='mse',
          metrics=['mae', 'acc'])


  def fit(self, *args, **kwargs):
    self.model.fit(*args, **kwargs)



def get_predictive_model_one_hot(intervened_state_logits, model_type):
  # Assumes input_state is one-hot with shape (batch, dims, one_hot_state)
  input_shape = output_shape = keras.backend.int_shape(intervened_state_logits)

  if model_type == 'fc':
    # Hard coded dimensions for now.

    # Assume for now that there is no covariance matrix
    assert len(input_shape) >= 2, 'wrong input shape'
    assert len(output_shape) >= 2, 'wrong output shape'

    output_dim = np.prod(input_shape[1:])

    # Hard coded dimensions for now.
    prediction_model = keras.models.Sequential([
      keras.layers.Flatten(name='flatten'),
      # keras.layers.Dense(32, input_shape=input_shape, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
      # keras.layers.Activation('relu'),
      keras.layers.Dense(output_dim, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name='fc1'),
      keras.layers.Reshape(output_shape[1:], name='flatten'),
    ], name='predictive_model')
  else:
    raise NotImplementedError('Model type %s is not supported' % model_type)


  output = keras.layers.add(
    [prediction_model(intervened_state_logits), intervened_state_logits],
    name='predicted_next_state_logits')
  return output
