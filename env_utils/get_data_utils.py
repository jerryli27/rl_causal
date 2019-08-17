import gym
import numpy as np
from tensorflow import keras

from data_structures import episode
from data_structures import step_result


def get_random_action(num_actions, env=None, env_name=None, const_action=None):
  if env is None:
    assert env_name, 'must provide either env or env_name'
    env = gym.make(env_name)
  ret = []
  if const_action is None:
    for _ in range(num_actions):
      action = env.action_space.sample()
      ret.append(action)
  else:
    ret = [const_action for _ in range(num_actions)]
  ret = np.array(ret)
  return ret

def get_data(env_name, num_episodes, env=None, const_action=None, add_random_actions=False):
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

  x, y = get_x_y_from_episodes(episodes)
  if add_random_actions:
    x['random_action'] = get_random_action(env=env, num_actions=x['action'].shape[0], const_action=const_action)
  return x, y

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


def get_x_y_from_episodes(episodes):
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
  filter = x['action'][:, 0] == action
  for key in x.keys():
    x_return[key] = x[key][filter]
  y_return = y_return[filter]
  return x_return, y_return

def add_batch_dim(dictinary):
  ret = {}
  for k in dictinary.keys():
    ret[k] = np.expand_dims(dictinary[k], axis=0)
  return ret

def convert_env_observation(observation, add_batch_dim=False):
  def maybe_add_batch_dim(x):
    if add_batch_dim:
      return np.expand_dims(x, axis=0)
    else:
      return x
  return {
    'current_state': maybe_add_batch_dim(observation['observation']),
    'goal': maybe_add_batch_dim(observation['desired_goal']),
  }


def combine_obs_dicts(obs_dicts):
  """Given a list of dictionaries, stack all their values."""
  assert obs_dicts, 'empty obs_dicts'
  ret = {}
  for k in obs_dicts[0].keys():
    v = [obs[k] for obs in obs_dicts]
    ret[k] = np.array(v)
  return ret