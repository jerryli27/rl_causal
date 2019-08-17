
from tensorflow import keras
import numpy as np
import gym

from causal_models import intervention_model
from causal_models import mine_exp
from causal_models import predictive_model
from env_utils import get_data_utils
from nn_utils import prob_utils
from nn_utils import vis_utils
from rl_models import critic_network
from rl_models import policy_network
from rl_models import ppo
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
    'current_state': x['current_state'][index],
    'action': x['action'][index],
    'intervened_state_one_hot': eval_output[0][index],
    'predicted_next_state_one_hot': eval_output[1][index],
    'actual_next_state': y[index],
  }
  print(debug_info)


def identity_loss(y_true, y_pred):
  # Maximize.
  return keras.backend.mean(y_pred)

def neg_identity_loss(y_true, y_pred):
  # Maximize.
  return - keras.backend.mean(y_pred)

if __name__ == '__main__':
  env = gym.make(ENV_NAME)

  if not isinstance(env.observation_space, gym.spaces.Dict):
    raise NotImplementedError('Only goal oriented envs are supported.')

  state_shape = env.observation_space['observation'].shape
  state_max_num_classes = np.max(env.observation_space['observation'].nvec)
  action_shape = env.action_space.shape
  num_possible_actions = env.action_space.nvec[0]  # TODO: support multiple dim actions.
  action_prob_shape = [1, num_possible_actions]
  model_type = 'fc'

  current_state = keras.layers.Input(shape=state_shape, dtype='int32', name='current_state')
  goal = keras.layers.Input(shape=state_shape, dtype='int32', name='goal')
  action = keras.layers.Input(shape=action_shape, dtype='int32', name='action')
  random_action = keras.layers.Input(shape=action_shape, dtype='int32', name='random_action')
  # action_prob_input is used for off-line learning
  # Assumes single action dim for now.
  sampling_action_prob = keras.layers.Input(shape=action_prob_shape, dtype='float', name='sampling_action_prob')
  reward_input = keras.layers.Input(shape=[1], dtype='float', name='reward')

  current_state_one_hot = keras.backend.one_hot(current_state, state_max_num_classes)
  goal_one_hot = keras.backend.one_hot(goal, state_max_num_classes)
  action_one_hot = keras.backend.one_hot(action, num_possible_actions)
  random_action_one_hot = keras.backend.one_hot(random_action, num_possible_actions)

  # TODO: make sure the intervention model supports multidim one hot.
  intervened_state_one_hot = intervention_model.get_intervention_model_one_hot(
    current_state_one_hot, action, num_possible_actions)
  predicted_next_state_logits = predictive_model.get_predictive_model_one_hot(intervened_state_one_hot, model_type)
  predicted_next_state_one_hot = keras.layers.Softmax()(predicted_next_state_logits)
  model = keras.models.Model(inputs=[current_state, action], outputs=[predicted_next_state_logits])

  optimizer = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None)
  model.compile(
    optimizer=optimizer,
    # loss=keras.losses.CategoricalCrossentropy(from_logits=False),  # , label_smoothing=0.05
    # metrics=[keras.metrics.CategoricalCrossentropy(from_logits=False)])
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # , label_smoothing=0.05
    metrics=[keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])
  # Generate dummy data

  train_data_size = 100
  test_data_size = train_data_size // 10

  # const_action=[0],
  x_train, y_train = get_data_utils.get_data(ENV_NAME, train_data_size, add_random_actions=True)
  x_test, y_test = get_data_utils.get_data(ENV_NAME, test_data_size, add_random_actions=True)
  x_eval, y_eval = get_data_utils.get_data(ENV_NAME, test_data_size, add_random_actions=True)

  # x_train, y_train = get_data_utils.get_x_y_from_episodes(train_episodes)
  # x_test, y_test = get_data_utils.get_x_y_from_episodes(test_episodes)
  # x_eval, y_eval = get_data_utils.get_x_y_from_episodes(eval_episodes)

  model.fit(x_train, y_train, epochs=10, batch_size=32)
  score = model.evaluate(x_test, y_test, batch_size=32)
  print('test score: ', score)
  eval_score = model.evaluate(x_eval, y_eval, batch_size=32)
  print('eval score: ', eval_score)
  x_test_chosen, y_test_chosen = get_data_utils.filter_by_action(x_test, y_test, action=0)
  test_chosen_score = model.evaluate(x_test_chosen, y_test_chosen, batch_size=32)
  print('test_chosen score: ', test_chosen_score)


  # Train a MINE model on the intervention model output.
  # # Get MINE training data.
  # intervened_state_one_hot_fn = keras.backend.function(
  #   inputs=[current_state, action], outputs=[intervened_state_one_hot])
  # intervened_state_one_hot_output_train = intervened_state_one_hot_fn(x_train['current_state'], x_train['action'])

  # Build MINE
  # mine_t_actual, mine_t_random = mine_exp.get_t_network(
  #   current_state_one_hot, action_one_hot, intervened_state_one_hot, random_action_one_hot)
  mine_t_actual, mine_t_random = mine_exp.get_t_network(
    current_state_one_hot, action_one_hot, predicted_next_state_one_hot, random_action_one_hot)
  mine_v = mine_exp.get_v_network(mine_t_actual, mine_t_random)

  # Train MINE
  mine_model = keras.models.Model(inputs=[current_state, action, random_action], outputs=[mine_v])
  for layer in mine_model.layers:
    if not layer.name.startswith('mine'):
      layer.trainable=False

  # optimizer = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None)
  mine_model.compile(
    optimizer=optimizer,
    loss=neg_identity_loss,  # , label_smoothing=0.05
    # metrics=[keras.metrics.SparseCategoricalCrossentropy(from_logits=True)]
  )

  mine_model.fit(x_train, y_train, epochs=10, batch_size=32)
  score = mine_model.evaluate(x_test, y_test, batch_size=32)
  print('test score: ', score)
  x_test_v_values = mine_model.predict_on_batch(x_test)
  print('x_test_v_values: ', x_test_v_values)


  for layer in mine_model.layers:
    if layer.name.startswith('mine'):
      mine_layer_weights = layer.get_weights()
      print('mine_layer_weights l1: ', np.mean(np.abs(mine_layer_weights)))

  mine_t_model = keras.models.Model(inputs=[current_state, action, random_action], outputs=[intervened_state_one_hot, mine_t_actual, mine_t_random, predicted_next_state_one_hot])
  mine_t_model.compile(
    optimizer=optimizer,
    loss=neg_identity_loss,
  )
  x_test_intervened_state, x_test_t_actual_values, x_test_t_random_values, x_test_predicted_next_state = mine_t_model.predict_on_batch(x_test)
  x_test_keys = [k for k in x_test.keys()]
  for i in range(min(10, x_test[x_test_keys[0]].shape[0])):
    print(i)
    for k in x_test_keys:
      print('x_test %s: ' %k, x_test[k][i])
    print('x_test_intervened_state: ', x_test_intervened_state[i])
    print('x_test_predicted_next_state: ', x_test_predicted_next_state[i])
    print('x_test_t_actual_values: ', x_test_t_actual_values[i])
    print('x_test_t_random_values: ', x_test_t_random_values[i])

  # Train PPO for each option.
  critic = critic_network.get_critic_network(current_state_one_hot, goal_one_hot)
  policy = policy_network.get_policy_network(current_state_one_hot, goal_one_hot, action_prob_shape)
  policy_loss = ppo.get_ppo_loss(policy, critic, reward_input, sampling_action_prob, action_one_hot)
  policy_fn = keras.backend.function(inputs=[current_state, goal], outputs=policy)

  # critic_model = keras.models.Model(inputs=[current_state, goal], outputs=critic)
  # critic_model.compile(
  #   optimizer=optimizer,
  #   loss='mse',
  # )
  # policy_loss_model = keras.models.Model(inputs=[current_state, goal, action, sampling_action_prob, reward_input], outputs=policy_loss)
  # policy_loss_model.compile(
  #   optimizer=optimizer,
  #   loss=identity_loss,
  # )

  joint_ac_loss_model = keras.models.Model(
    inputs=[current_state, goal, action, sampling_action_prob, reward_input],
    outputs=[critic, policy_loss])
  joint_ac_loss_model.compile(
    optimizer=optimizer,
    loss={'tf_op_layer_Neg_1': identity_loss, 'critic': 'mse'},
    loss_weights={'tf_op_layer_Neg_1': 1., 'critic': 1.0}
  )

  ppo_num_epochs = 1000
  ppo_num_data_points = 32
  for _ in range(ppo_num_epochs):
    ppo_batch_x, ppo_batch_y = ppo.get_batch(policy_fn, env, ppo_num_data_points, is_continuous=False, is_eval=False)

    # Train policy and critic
    # policy_loss_model.fit(ppo_batch_x, ppo_batch_y, epochs=10, batch_size=32)
    # critic_model.fit(ppo_batch_x, ppo_batch_y, epochs=10, batch_size=32)
    joint_ac_loss_model.fit(ppo_batch_x, ppo_batch_y, epochs=1, batch_size=32)

  ppo_batch_x_test, ppo_batch_y_test = ppo.get_batch(policy_fn, env, ppo_num_data_points, is_continuous=False,
                                                     is_eval=False)

  # policy_score = policy_loss_model.evaluate(ppo_batch_x_test, ppo_batch_y_test, batch_size=32)
  # critic_score = critic_model.evaluate(ppo_batch_x_test, ppo_batch_y_test, batch_size=32)
  # ppo_batch_predicted_y_test = critic_model.predict(ppo_batch_x_test, batch_size=32)


  # _, policy_score, critic_score = joint_ac_loss_model.evaluate(ppo_batch_x_test, ppo_batch_y_test, batch_size=32)
  # _, ppo_batch_predicted_y_test = joint_ac_loss_model.predict(ppo_batch_x_test, batch_size=32)
  # print('policy_score: %.3f  critic score: %.3f' %(policy_score, critic_score))
  print('done')


  # # Examine model weights
  # weights_dict = vis_utils.get_weights_with_name(model)
  # for name, weight in weights_dict.items():
  #   print(name, weight.shape, weight)
  #   if name == 'predictive_model/fc1/kernel:0':
  #     print('Kernel 10th vector: ', weight[10])
  #     print('Kernel 10th vector expected: ', [0] * 11 + [1] + [0] * 8)
  #     print('Kernel 11th vector: ', weight[11])
  #     print('Kernel 11th vector expected: ', [0] * 10 + [1] + [0] * 9)
  #
  # # Examine intervened_state_logits
  #
  # # prediction_debug_model = keras.models.Model(
  # #   inputs=[current_state, action],
  # #   outputs=[intervened_state_one_hot, predicted_next_state_one_hot])
  # # prediction_debug_model.compile(
  # #   optimizer=optimizer,)
  # # prediction_debug_model_output = prediction_debug_model.predict(x_test)
  # for _ in range(10):
  #   model.fit(x_train, y_train, epochs=1, batch_size=32)
  #   debug_predict = keras.backend.function(inputs=[current_state, action],
  #     outputs=[intervened_state_one_hot, predicted_next_state_one_hot])
  #   prediction_debug_model_output=debug_predict([x_test['current_state'], x_test['action']])
  #   print_debug_info(x_test, y_test, eval_output=prediction_debug_model_output, index=0)
  #
  #
  # # action_t_test = intervention_model.infer_action(intervened_state_one_hot, current_state, current_state_one_hot, action, x_eval, p_value_threshold=0.05)
  # # print(action_t_test)
  #
  # # # Evaluate on one specific action.
  # for chosen_action in range(num_possible_actions):
  #   x_test_chosen, y_test_chosen = get_data_utils.filter_by_action(x_test, y_test, action=chosen_action)
  #   action_t_test = intervention_model.infer_action(intervened_state_one_hot, current_state, current_state_one_hot, action, x_test, x_test_chosen, p_value_threshold=0.05)
  #   print('T test for action %d' %chosen_action)
  #   print(action_t_test.pvalue)
  #
  # # TODO: but how do you evaluate the causal graph of intervened_s->s'?  Given so many dims, the possibilities are too many.
  #
  # print('Done')
