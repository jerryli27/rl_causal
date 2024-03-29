import gym
import numpy as np
from tensorflow import keras

from data_structures import episode
from data_structures import step_result

# OBSERVATION_KEYS = ('state', 'next_state', 'goal', 'reward')

def _get_x_from_input(action_input):
  ret = {k: [] for k in action_input.keys()}
  ret.update({'next_'+k: [] for k in action_input.keys()})
  return ret

def _get_empty_x(state_input, action_input):
  x = {'reward': []}
  state_x = _get_x_from_input(state_input)
  x.update(state_x)
  action_x = _get_x_from_input(action_input)
  x.update(action_x)
  return x



def _convert_observation_to_x(x, observation, next_observation, state_input):
  for i, (obs_type, v) in enumerate(state_input.items()):
    assert obs_type.startswith('state_')

    env_obs_type = obs_type[len('state_'):]
    obs_val = observation[env_obs_type]
    x[obs_type].append(obs_val)

    next_obs_type = 'next_' + obs_type
    next_obs_val = next_observation[env_obs_type]
    x[next_obs_type].append(next_obs_val)

def _convert_action_to_x(x, action, action_input):
  action_type_i, action_val = action
  x['action_type'].append(action_type_i)
  for i, (k, v) in enumerate(action_input.items()):

    action_type = k
    if action_type == 'action_type':
      continue
    if i == action_type_i:
      x[action_type].append(action_val)
    else:
      x[action_type].append(np.zeros(v.shape[1:], v.dtype.as_numpy_dtype))


def _convert_action_info_to_x(x, action_info):
  assert len(action_info) == 3
  def maybe_create_and_append(name, data):
    if name not in x:
      x[name] = []
    x[name].append(data)

  maybe_create_and_append('sampling_action_mean', action_info[0])
  maybe_create_and_append('sampling_action_var', action_info[1])
  maybe_create_and_append('sampling_action_embed', action_info[2])

def get_random_action(num_actions, action_input, env=None, env_name=None, const_action=None):
  if env is None:
    assert env_name, 'must provide either env or env_name'
    env = gym.make(env_name)

  random_action_x = _get_x_from_input(action_input)
  if const_action is None:
    for _ in range(num_actions):
      action = env.action_space.sample()
      _convert_action_to_x(random_action_x, action, action_input)
  else:
    raise NotImplementedError
    # ret = [const_action for _ in range(num_actions)]

  ret = {}
  for k in random_action_x.keys():
    ret['random_' + k] = np.array(random_action_x[k])
  return ret


def get_data(num_episodes, state_input, action_input, gamma, env_name=None, env=None, get_action_fn=None, add_random_actions=True, use_augmented_goal=False, render=False):
  if env is None:
    env = gym.make(env_name)
  assert env is not None, 'Please provide either env or env_name.'
  if get_action_fn is None:
    get_action_fn = lambda _: (env.action_space.sample(), None)
  episodes = []
  def maybe_render_env():
    if render:
      env.render()

  for _ in range(num_episodes):
    init_state = env.reset()
    maybe_render_env()
    obs = init_state
    done = False
    step_results = []
    while not done:
      action, action_info = get_action_fn(obs)
      obs, reward, done, info = env.step(action)
      maybe_render_env()
      step_results.append(step_result.StepResult(action, obs, reward, done, info, action_info=action_info))
    episodes.append(episode.Episode(init_state, step_results, gamma=gamma))

  x, y = get_x_y_from_episodes(episodes, state_input, action_input, use_augmented_goal=use_augmented_goal, env=env)
  if add_random_actions:
    x.update(get_random_action(env=env, action_input=action_input, num_actions=x['state_observation'].shape[0], ))
  return x, y


def get_x_y_from_one_episode(episode, state_input, action_input, use_augmented_goal=False, env=None):
  x = _get_empty_x(state_input, action_input)
  y = []
  if use_augmented_goal:
    assert ('next_state_desired_goal' in x) and ('next_state_reward' in x)

  for i, step_result in enumerate(episode.step_results):
    obs = episode.init_state if i == 0 else episode.step_results[i-1].obs
    next_obs = episode.step_results[i].obs
    x['reward'].append(episode.step_results[i].reward)
    y.append(step_result.obs['observation'])

    _convert_observation_to_x(x, obs, next_obs, state_input)
    if use_augmented_goal:
      # Replace the original goal with goal = next_state.
      next_state = goal = next_obs['observation']
      reward = env.compute_reward(next_state, goal, info=False)
      x['next_state_desired_goal'][-1] = goal
      x['next_state_reward'][-1] = reward

    # Actions
    _convert_action_to_x(x, step_result.action, action_input)
    if step_result.action_info is not None:
      _convert_action_info_to_x(x, step_result.action_info)

  return x, y


def get_x_y_from_episodes(episodes, state_input, action_input, use_augmented_goal=False, env=None):
  assert episodes
  x = None
  y = []
  for episode in episodes:
    curr_xs, curr_ys = get_x_y_from_one_episode(episode, state_input, action_input, use_augmented_goal=use_augmented_goal, env=env)
    if x is None:
      x = {k: [] for k in curr_xs.keys()}
    for k in curr_xs.keys():
      x[k].extend(curr_xs[k])
    y.extend(curr_ys)

  for k in x.keys():
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
  if isinstance(data, list) or isinstance(data, tuple):
    ret = []
    for v in data:
      if assert_batch_size_is_one:
        assert v.shape[0] == 1, 'batch size is not 1 but %d' %(v.shape[0])
      ret.append(v[0])
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
  return {'state_' + k: maybe_add_batch_dim(v) for k, v in observation.items()}


def combine_obs_dicts(obs_dicts):
  """Given a list of dictionaries, stack all their values."""
  assert obs_dicts, 'empty obs_dicts'
  ret = {}
  for k in obs_dicts[0].keys():
    v = [obs[k] for obs in obs_dicts]
    ret[k] = np.array(v)
  return ret