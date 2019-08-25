"""Creates a wrapper around a gym environment. Allows the user to add custom pretrained options."""
import copy

import gym
from env_utils.spaces import me_dict_utils

NATIVE = 'native'

class EnvRecordLastActionWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.last_action = None

  def reset(self, **kwargs):
    self.last_action = None
    return self.env.reset(**kwargs)

  def step(self, action):
    self.last_action = action
    return self.env.step(action)

  def render(self, mode='human', **kwargs):
    print('last_action:', self.last_action)
    return super().render(mode='human', **kwargs)

class GymEnvWrapper(gym.Wrapper):
  def __init__(self, env, is_eval=False):
    super(GymEnvWrapper, self).__init__(env)
    self.env = env  # type: gym.Env

    self.options = {}
    self.is_eval = is_eval
    self._curr_observation = None

    # Properties for the gym env.
    self.action_space = me_dict_utils.MutuallyExclusiveDict({NATIVE: self.env.action_space, })
    self.observation_space = self.env.observation_space
    self.reward_range = self.env.reward_range

    self.reset()

  def _construct_action_space(self):
    action_dict = {NATIVE: self.env.action_space, }
    for k, v in self.options.items():
      action_dict[k] = v.goal_space
    self.action_space = me_dict_utils.MutuallyExclusiveDict(action_dict)

  def _action_is_option(self, action):
    return action[0] != 0  # Assume 0 is always native.

  def add_option(self, option):
    self.options[option.name] = option
    self._construct_action_space()
    raise NotImplementedError('verify tjis works')

  def step(self, action):
    if self._curr_observation is None:
      raise AssertionError
    if not self.action_space.contains(action):
      raise ValueError('Action must be legal.')
    if self._action_is_option(action):
      option_name = action[0]
      goal = action[1]
      # TODO: I need to set the option's goal... I also need to pass in the current observation.
      ret = self.options[option_name].run_until_termination(
        observation=self._curr_observation, goal=goal, is_eval=self.is_eval)
    else:
      ret = self.env.step(action[1])

    observation = ret[0]  # ret= (observation, reward, done, info.)
    self._curr_observation = copy.deepcopy(observation)
    return ret

  def reset(self):
    observation = self.env.reset()
    self._curr_observation = copy.deepcopy(observation)
    return observation