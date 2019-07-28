import gym
import numpy as np
from tensorflow import keras

from data_structures import episode
from data_structures import step_result


def get_data(env_name, num_episodes, env=None, const_action=None):
  if env is None:
    env = gym.make(env_name)
  episodes = []

  for _ in range(num_episodes):
    init_state = env.reset()
    done = False
    step_results = []
    while not done:
      if const_action is None:
        action = env.action_space.sample()
      else:
        action = const_action
      obs, reward, done, info = env.step(action)
      step_results.append(step_result.StepResult(action, obs, reward, done, info))
    episodes.append(episode.Episode(init_state, step_results))
  return episodes


def get_init_data(env_name, num_episodes):
  env = gym.make(env_name)
  noop_action = env.get_noop_action()
  return get_data(env_name, num_episodes, env=env, const_action=noop_action)


def get_x_y_from_one_episode(episode):
  x = {
    'current_state': [],
    'action': [],
  }
  y = []
  for i, step_result in enumerate(episode.step_results):
    current_state = episode.init_state if i == 0 else episode.step_results[i-1].obs
    current_state = current_state['observation']
    x['current_state'].append(current_state)
    x['action'].append(step_result.action)
    y.append(step_result.obs['observation'])
  return x, y


def get_x_y_from_episodes(episodes, convert_y_one_hot=False):
  assert episodes
  x = {
    'current_state': [],
    'action': [],
  }
  y = []
  for episode in episodes:
    curr_xs, curr_ys = get_x_y_from_one_episode(episode)
    x['current_state'].extend(curr_xs['current_state'])
    x['action'].extend(curr_xs['action'])
    y.extend(curr_ys)

  x['current_state'] = np.array(x['current_state'])
  x['action'] = np.array(x['action'])
  y = np.array(y)
  return x, y


def filter_by_action(x, y, action):
  x_return = {}
  y_return = np.copy(y)
  filter = x['action'] == action
  for key in x.keys():
    x_return[key] = x[key][filter]
  y_return = y_return[filter]
  return x_return, y_return
