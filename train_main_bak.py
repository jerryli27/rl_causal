
from tensorflow import keras
import numpy as np

from causal_models import intervention_model
from causal_models import predictive_model


def get_action_embed(state_shape, num_possible_actions):
  return np.random.random((num_possible_actions, state_shape[0]))

def get_data(size, action_embed, state_shape, num_possible_actions):
  x = {
    'current_state': np.random.random((size, state_shape[0])),
    'action': np.random.randint(0, num_possible_actions, (size, action_shape[0])),
  }
  one_hot_action = np.eye(num_possible_actions)[x['action']]
  one_hot_action = one_hot_action[:, 0, :]  # Get rid of the extra dim
  y = x['current_state'] + np.matmul(one_hot_action, action_embed)
  return x, y

def print_action_embed_diff(model, action_embed):
  predicted_action_embed_layer = model.get_layer('action_embed')
  predicted_action_embed_weights = predicted_action_embed_layer.get_weights()
  print_diff(predicted_action_embed_weights, action_embed)

def print_diff(predicted, expected):
  print(predicted)
  print(expected)
  diff = np.average(np.abs(predicted - expected))
  print(diff)

if __name__ == '__main__':
  state_shape = (10,)
  action_shape = (1,)
  num_possible_actions = 5
  # output_shape = (10,)
  model_type = 'fc'
  # intervention_m = intervention_model.InterventionModel(input_shape, model_type)
  # model = predictive_model.PredictiveModel(input_shape, output_shape, model_type, intervention_m)

  current_state = keras.layers.Input(shape=state_shape, name='current_state')
  action = keras.layers.Input(shape=action_shape, dtype='int32', name='action')

  intervened_state = intervention_model.get_intervention_model_one_hot(current_state, action, num_possible_actions)
  predicted_next_state = predictive_model.get_predictive_model_one_hot(intervened_state, model_type)
  model = keras.models.Model(inputs=[current_state, action], outputs=[predicted_next_state])

  optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
  model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae', 'acc'])
  # Generate dummy data

  train_data_size = 10000
  test_data_size = train_data_size // 10

  action_embed = get_action_embed(state_shape, num_possible_actions)
  x_train , y_train = get_data(train_data_size, action_embed, state_shape, num_possible_actions)
  x_test , y_test = get_data(test_data_size, action_embed, state_shape, num_possible_actions)
  print_action_embed_diff(model, action_embed)

  model.fit(x_train, y_train, epochs=10, batch_size=32)
  score = model.evaluate(x_test, y_test, batch_size=32)
  print(score)

  print_action_embed_diff(model, action_embed)

  intervened_state_layer = model.get_layer('intervened_state')
  intervened_state_model = keras.models.Model(inputs=model.input,
                                          outputs=intervened_state_layer.output)
  intervened_state_output = intervened_state_model.predict(x_test)
  print_diff(intervened_state_output, y_test)

  model.summary()

  fc1_layer = model.get_layer('predictive_model').get_layer('fc1')
  fc1_weights, fc1_biases = fc1_layer.get_weights()
  fc1_weights = np.reshape(fc1_weights, [-1])
  print('fc1_weights mean, variance, max, min', np.average(fc1_weights), np.var(fc1_weights), np.max(fc1_weights), np.min(fc1_weights))

  print('Done')