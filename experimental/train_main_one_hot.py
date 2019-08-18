"""Reason this is abandoned: the loss did not really work. It got larger over time, which leads me to rethink my
approach to convert one-hot states into label-smoothed logits, operates on logits internally in the model, and convert
it back at the very last stage. Perhaps I should stop treating states like they must be one-hot. Perhaps there
should be no softmax.

"""
from tensorflow import keras
import numpy as np
import gym

from causal_models import intervention_model_lib
from causal_models import predictive_model
from env_utils import get_data_utils
from nn_utils import prob_utils
from nn_utils import vis_utils
import custom_envs

ENV_NAME = 'TwoDigits-v0'
assert custom_envs


def print_diff(predicted, expected):
  print(predicted)
  print(expected)
  diff = np.average(np.abs(predicted - expected))
  print(diff)


def print_debug_info(x, y, eval_output, index):
  debug_info = {
    'state_input': x['state_input'][index],
    'action_input': x['action_input'][index],
    'intervened_state_embed': eval_output[0][index],
    'predicted_next_state_embed': eval_output[1][index],
    'actual_next_state': y[index],
  }
  print(debug_info)

if __name__ == '__main__':
  env = gym.make(ENV_NAME)

  if not isinstance(env.observation_space, gym.spaces.Dict):
    raise NotImplementedError('Only goal_input oriented envs are supported.')

  state_shape = env.observation_space['observation'].shape
  state_max_num_classes = np.max(env.observation_space['observation'].nvec)
  action_shape = env.action_space.shape
  num_possible_actions = env.action_space.n
  model_type = 'fc'

  current_state = keras.layers.Input(shape=state_shape, dtype='int32', name='state_input')
  action = keras.layers.Input(shape=action_shape, dtype='int32', name='action_input')

  current_state_one_hot = keras.backend.one_hot(current_state, state_max_num_classes)
  current_state_one_hot_smoothed = prob_utils.smooth_one_hot(current_state_one_hot, label_smoothing=0.05)
  current_state_logits = prob_utils.inverse_softmax(current_state_one_hot_smoothed)

  # TODO: make sure the intervention model supports multidim one hot.
  intervened_state_logits = intervention_model_lib.get_intervention_model_one_hot(
    current_state_logits, action, num_possible_actions)
  predicted_next_state_logits = predictive_model.get_predicted_next_state(intervened_state_logits, model_type)
  predicted_next_state_one_hot = keras.layers.Softmax(name='predicted_next_state_embed')(predicted_next_state_logits)
  model = keras.models.Model(inputs=[current_state, action], outputs=[predicted_next_state_logits])

  optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
  model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05),
    metrics=[keras.metrics.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05)])
  # Generate dummy data

  train_data_size = 1000
  test_data_size = train_data_size // 10

  train_episodes = get_data_utils.get_init_data(ENV_NAME, train_data_size)
  test_episodes = get_data_utils.get_init_data(ENV_NAME, test_data_size)
  eval_episodes = get_data_utils.get_data(ENV_NAME, test_data_size)

  x_train, y_train = get_data_utils.get_x_y_from_episodes(train_episodes)
  x_test, y_test = get_data_utils.get_x_y_from_episodes(test_episodes)
  x_eval, y_eval = get_data_utils.get_x_y_from_episodes(eval_episodes)
  y_train = keras.utils.to_categorical(y_train, num_classes=10)
  y_test = keras.utils.to_categorical(y_test, num_classes=10)
  y_eval = keras.utils.to_categorical(y_eval, num_classes=10)

  model.fit(x_train, y_train, epochs=10, batch_size=32)
  score = model.evaluate(x_test, y_test, batch_size=32)
  print('test score: ', score)
  eval_score = model.evaluate(x_eval, y_eval, batch_size=32)
  print('eval score: ', eval_score)
  x_test_chosen, y_test_chosen = get_data_utils.filter_by_action(x_test, y_test, action=0)
  test_chosen_score = model.evaluate(x_test_chosen, y_test_chosen, batch_size=32)
  print('test_chosen score: ', test_chosen_score)

  # Examine model weights
  weights_dict = vis_utils.get_weights_with_name(model)
  for name, weight in weights_dict.items():
    print(name, weight.shape, weight)
    if name == 'predictive_model/fc1/kernel:0':
      print('Kernel 10th vector: ', weight[10])
      print('Kernel 10th vector expected: ', [0] * 11 + [1] + [0] * 8)
      print('Kernel 11th vector: ', weight[11])
      print('Kernel 10th vector expected: ', [0] * 10 + [1] + [0] * 9)

  # Examine intervened_state

  intervened_state_one_hot = keras.layers.Softmax(name='intervened_state_embed')(intervened_state_logits)
  # prediction_debug_model = keras.models.Model(
  #   inputs=[state_input, action_input],
  #   outputs=[intervened_state_embed, predicted_next_state_embed])
  # prediction_debug_model.compile(
  #   optimizer=optimizer,)
  # prediction_debug_model_output = prediction_debug_model.predict(x_test)
  for _ in range(10):
    model.fit(x_train, y_train, epochs=1, batch_size=32)
    debug_predict = keras.backend.function(inputs=[current_state, action],
      outputs=[intervened_state_one_hot, predicted_next_state_one_hot])
    prediction_debug_model_output=debug_predict([x_test['state_input'], x_test['action_input']])
    print_debug_info(x_test, y_test, eval_output=prediction_debug_model_output, index=0)






  action_t_test = intervention_model_lib.infer_action(intervened_state_one_hot, current_state, current_state_one_hot_smoothed, action, x_eval, p_value_threshold=0.05)
  print(action_t_test)

  # Evaluate on one specific action_input.
  action_t_test = intervention_model_lib.infer_action(intervened_state_one_hot, current_state, current_state_one_hot_smoothed, action, x_test_chosen, p_value_threshold=0.05)
  print(action_t_test)
  print('Done')
