import gym
import numpy as np
from tensorflow import keras

from data_structures import episode
from data_structures import step_result

OBSERVATION_KEYS = ('state', 'action', 'next_state')

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

def get_data(num_episodes, env_name=None, env=None, const_action=None, add_random_actions=True):
  if env is None:
    env = gym.make(env_name)
  assert env is not None, 'Please provide either env or env_name.'
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
  x = {k: [] for k in OBSERVATION_KEYS}
  y = []
  for i, step_result in enumerate(episode.step_results):
    current_state = episode.init_state if i == 0 else episode.step_results[i-1].obs
    current_state = current_state['observation']
    next_state = episode.step_results[i].obs['observation']

    x['state'].append(current_state)
    # TODO: is it possible to have one unified x and y for all the models?
    raise NotImplementedError
    x['action'].append(step_result.action)
    x['next_state'].append(next_state)
    y.append(step_result.obs['observation'])
  return x, y


def get_x_y_from_episodes(episodes):
  raise NotImplementedError('Does not support multi action yet.')
  assert episodes
  x = {k: [] for k in OBSERVATION_KEYS}
  y = []
  for episode in episodes:
    curr_xs, curr_ys = get_x_y_from_one_episode(episode)
    for k in OBSERVATION_KEYS:
      x[k].extend(curr_xs[k])
    y.extend(curr_ys)

  for k in OBSERVATION_KEYS:
    x[k] = np.array(x[k])
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


def replace_goal_in_observation(observation, goal):
  assert 'goal' in observation
  observation['goal'] = goal


def add_batch_dim(dictinary):
  ret = {}
  for k in dictinary.keys():
    ret[k] = np.expand_dims(dictinary[k], axis=0)
  return ret

def remove_batch_dim(data, assert_batch_size_is_one=False):
  if isinstance(data, dict):
    ret = {}
    for k in data.keys():
      if assert_batch_size_is_one:
        assert data[k].shape[0] == 1, 'batch size is not 1 but %d' %(data[k].shape[0])
      ret[k] = data[k][0]
  elif hasattr(data, 'shape'):
    if assert_batch_size_is_one:
      assert data.shape[0] == 1, 'batch size is not 1 but %d' %(data.shape[0])
    ret = data[0]
  else:
    raise NotImplementedError('Unsupported data type %s' %(str(type(data))))
  return ret

def convert_env_observation(observation, add_batch_dim=False):
  def maybe_add_batch_dim(x):
    if add_batch_dim:
      return np.expand_dims(x, axis=0)
    else:
      return x
  return {
    'state': maybe_add_batch_dim(observation['observation']),
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