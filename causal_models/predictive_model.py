"""A predictive model to model the state transition probabilities."""
import numpy as np
from tensorflow import keras


def get_predicted_next_state(intervened_state, output_activation_fn=None):
  # Assumes input_state is one-hot with shape (batch, types_dim, embed_dim)
  input_shape = output_shape = keras.backend.int_shape(intervened_state)

  # Hard coded dimensions for now.
  # Assume for now that there is no covariance matrix
  assert len(input_shape) >= 2, 'wrong input shape'
  assert len(output_shape) >= 2, 'wrong output shape'
  output_dim = np.prod(input_shape[1:])

  # Hard coded dimensions for now.
  prediction_model = keras.models.Sequential([
    keras.layers.Flatten(name='flatten'),
    keras.layers.Dense(output_dim, activation=None, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), name='fc1'),
    keras.layers.Reshape(output_shape[1:], name='flatten'),
  ], name='predictive_model')

  predicted_next_state_embed = keras.layers.add(
    [prediction_model(intervened_state), intervened_state],
    name='predicted_next_state_embed')
  if output_activation_fn is not None:
    predicted_next_state_embed = output_activation_fn(predicted_next_state_embed)
  return predicted_next_state_embed
