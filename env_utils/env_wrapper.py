"""Creates a wrapper around a gym environment. Allows the user to add custom pretrained options."""
import gym
from env_utils.spaces import me_dict

NATIVE = 'native'


class GymEnvWrapper(gym.Env):
  def __init__(self, env):
    self.env = env  # type: gym.Env
    self.options = {}

    # Properties for the gym env.
    self.action_space = me_dict.MutuallyExclusiveDict({NATIVE: self.env.action_space, })
    self.observation_space = self.env.observation_space
    self.reward_range = self.env.reward_range

  def _action_is_option(self, action):
    for k, v in action.items():
      if k == NATIVE:
        return True
    return False

  def add_option(self, option):
    raise NotImplementedError

  def seed(self, **kwargs):
    return self.env.seed(**kwargs)

  def step(self, action):
    if not isinstance(action, dict):
      raise ValueError('Action must be a dictionary.')
    if self._action_is_option(action):
      raise NotImplementedError
    else:
      return self.env.step(action[NATIVE])

  def reset(self):
    return self.env.reset()

  def render(self, **kwargs):
    return self.env.render(**kwargs)

  def close(self):
    return self.env.close()