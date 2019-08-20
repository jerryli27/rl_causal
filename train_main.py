
from tensorflow import keras
import numpy as np
import gym
import tensorflow as tf
from absl import app
from absl import flags

from causal_models import intervention_model_lib
from causal_models import mine_exp
from causal_models import predictive_model
from env_utils import env_rl_utils
from env_utils import env_wrapper
from env_utils import get_data_utils
from nn_utils import keras_utils
from rl_models import critic_network
from rl_models import policy_network
from rl_models import ppo
import custom_envs
assert custom_envs

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'TwoDigits-v0', 'Name of the gym environment.')


def main(unused_argv):
  env = env_wrapper.GymEnvWrapper(gym.make(FLAGS.env))
  if not isinstance(env.observation_space, gym.spaces.Dict):
    raise NotImplementedError('Only goal_input oriented envs are supported.')

  optimizer = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999)

  state_input, state_one_hot = env_rl_utils.get_input_from_space(env.observation_space['observation'], name='state')
  next_state_input, next_state_one_hot = env_rl_utils.get_input_from_space(env.observation_space['observation'], name='next_state')
  goal_input, goal_one_hot = env_rl_utils.get_input_from_space(env.observation_space['desired_goal'], name='goal')
  action_input, action_one_hot = env_rl_utils.get_input_from_space(env.action_space, name='action')
  random_action_input, random_action_one_hot = env_rl_utils.get_input_from_space(env.action_space, name='random_action')

  # TODO: action prob shape needs to change per action type.
  # sampling_action_prob = keras.layers.Input(shape=action_one_hot.shape[1:], dtype='float', name='sampling_action_prob')
  reward_input = keras.layers.Input(shape=[1], dtype='float', name='reward')

  # For now the embeddings are just the identity one hot embeddings.
  state_embed = state_one_hot
  goal_embed = goal_one_hot
  action_embed = action_one_hot  # TODO: add reversible action encoder/decoder.
  next_state_embed = next_state_one_hot
  random_action_embed = random_action_one_hot
  # _, s_types_dim, s_embed_dim = state_embed.shape
  # _, a_types_dim, a_embed_dim = action_embed.shape


  # raise NotImplementedError('Continue refactor from here.')
  # TODO(jryli): ('Add options and loop')
  # Generate data
  train_data_size = 100
  test_data_size = train_data_size // 10
  x_train, y_train = get_data_utils.get_data(num_episodes=train_data_size, env=env)
  x_test, y_test = get_data_utils.get_data(num_episodes=test_data_size, env=env)
  x_eval, y_eval = get_data_utils.get_data(num_episodes=test_data_size, env=env)

  # Train a MINE model on the intervention model output.
  # Build MINE
  mine_t_actual, mine_t_random = mine_exp.get_t_network(
    state_embed, action_embed, next_state_embed, random_action_embed)
  mine_v = mine_exp.get_v_network(mine_t_actual, mine_t_random)

  # Train MINE
  mine_model = keras.models.Model(
    inputs=[state_input, action_input, next_state_input, random_action_input],
    outputs=[mine_v])
  for layer in mine_model.layers:
    if not layer.name.startswith('mine'):
      layer.trainable = False

  mine_model.compile(optimizer=optimizer, loss=keras_utils.neg_identity_loss)
  mine_model.fit(x_train, y_train, epochs=10, batch_size=32)
  score = mine_model.evaluate(x_test, y_test, batch_size=32)
  print('test score: ', score)
  x_test_v_values = mine_model.predict_on_batch(x_test)
  print('x_test_v_values: ', x_test_v_values)
  causal_relation = (x_test_v_values > 0.5)  # TODO: hard coded threshold for now.

  # mine_t_model = keras.models.Model(inputs=[state_input, action_input, random_action], outputs=[intervened_state_embed, mine_t_actual, mine_t_random, predicted_next_state_embed])
  # mine_t_model.compile(
  #   optimizer=optimizer,
  #   loss=keras_utils.neg_identity_loss,
  # )
  # x_test_intervened_state, x_test_t_actual_values, x_test_t_random_values, x_test_predicted_next_state = mine_t_model.predict_on_batch(x_test)
  # x_test_keys = [k for k in x_test.keys()]
  # for i in range(min(10, x_test[x_test_keys[0]].shape[0])):
  #   print(i)
  #   for k in x_test_keys:
  #     print('x_test %s: ' %k, x_test[k][i])
  #   print('x_test_intervened_state: ', x_test_intervened_state[i])
  #   print('x_test_predicted_next_state: ', x_test_predicted_next_state[i])
  #   print('x_test_t_actual_values: ', x_test_t_actual_values[i])
  #   print('x_test_t_random_values: ', x_test_t_random_values[i])

  # *****************************
  # Train intervention model
  # *****************************
  intervention_model = intervention_model_lib.InterventionModel()
  # TODO(jryli): Can we train the MINE jointly with the rest? Or start with a fixed causal relation and create a new one every iteration? (take into consideration that there will be more options at the end of each loop)
  intervened_state_embed = intervention_model((state_embed, action_embed, causal_relation))
  # TODO(jryli): the loss and metrics should be gin-configurable. Also add state encoder.
  predicted_next_state_embed = predictive_model.get_predicted_next_state(intervened_state_embed,
                                                                         output_activation_fn=keras.layers.Softmax())

  predicted_next_state_model = keras.models.Model(
    inputs=[state_input, action_input], outputs=[predicted_next_state_embed])
  predicted_next_state_model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalCrossentropy(from_logits=False)])

  predicted_next_state_model.fit(x_train, y_train, epochs=10, batch_size=32)
  score = predicted_next_state_model.evaluate(x_test, y_test, batch_size=32)
  print('test score: ', score)
  eval_score = predicted_next_state_model.evaluate(x_eval, y_eval, batch_size=32)
  print('eval score: ', eval_score)
  # The following code tests how well we do on one specific action.
  # x_test_chosen, y_test_chosen = get_data_utils.filter_by_action(x_test, y_test, action=0)
  # test_chosen_score = predicted_next_state_model.evaluate(x_test_chosen, y_test_chosen, batch_size=32)
  # print('test_chosen score: ', test_chosen_score)

  # *****************************
  # Train PPO only for the ones with causal relations.
  # *****************************
  actor_critic_for_state_type = {}
  for state_type_i in range(s_types_dim):
    allowed_action_types = [action_type_i for action_type_i in range(len(causal_relation[state_type_i]))
                       if causal_relation[state_type_i][action_type_i]]
    if not allowed_action_types:
      continue
    critic = critic_network.get_critic_network(state_embed, goal_embed)
    policy = policy_network.get_policy_network(state_embed, goal_embed, allowed_action_types, env.action_space)
    policy_loss = ppo.get_ppo_loss(policy, critic, reward_input, allowed_action_types, sampling_action_prob, action_one_hot)
    policy_fn = keras.backend.function(inputs=[state_input, goal_input], outputs=policy)

    joint_ac_loss_model = keras.models.Model(
      inputs=[state_input, goal_input, action_input, sampling_action_prob, reward_input],
      outputs=[critic, policy_loss])
    joint_ac_loss_model.compile(
      optimizer=optimizer,
      loss={'ppo_loss': keras_utils.identity_loss, 'critic': 'mse'},
      loss_weights={'ppo_loss': 1., 'critic': 1.0}
    )

    ppo_num_epochs = 2
    ppo_num_data_points = 32
    for _ in range(ppo_num_epochs):
      ppo_batch_x, ppo_batch_y = ppo.get_batch(policy_fn, env, ppo_num_data_points, allowed_action_types=allowed_action_types, is_continuous=False, is_eval=False)
      # Train policy and critic
      joint_ac_loss_model.fit(ppo_batch_x, ppo_batch_y, epochs=1, batch_size=32)

    # ppo_batch_x_test, ppo_batch_y_test = ppo.get_batch(policy_fn, env, ppo_num_data_points, allowed_action_types=allowed_action_types,is_continuous=False, is_eval=False)
    actor_critic_for_state_type[state_type_i] = {
      'critic': critic,
      'policy': policy,
    }

  # policy_score = policy_loss_model.evaluate(ppo_batch_x_test, ppo_batch_y_test, batch_size=32)
  # critic_score = critic_model.evaluate(ppo_batch_x_test, ppo_batch_y_test, batch_size=32)
  # ppo_batch_predicted_y_test = critic_model.predict(ppo_batch_x_test, batch_size=32)


  # _, policy_score, critic_score = joint_ac_loss_model.evaluate(ppo_batch_x_test, ppo_batch_y_test, batch_size=32)
  # _, ppo_batch_predicted_y_test = joint_ac_loss_model.predict(ppo_batch_x_test, batch_size=32)
  # print('policy_score: %.3f  critic score: %.3f' %(policy_score, critic_score))
  print('done')

if __name__ == '__main__':
  app.run(main)