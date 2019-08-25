"""Option class for hierarchical RL."""

from env_utils import env_rl_utils
from env_utils import get_data_utils
from env_utils.spaces import me_dict_utils
from nn_utils import policy_utils


def never_terminate_fn(unused_state):
  return False


class Option(object):

  def __init__(self, policy_fn, env, allowed_action_types, goal_space, termination_fn=never_terminate_fn, name=''):
    self.policy_fn = policy_fn
    # Given the state, which contains the goal, returns True if the option should stop.
    self.termination_fn = termination_fn
    self.env = env
    assert isinstance(self.env.action_space, me_dict_utils.MutuallyExclusiveDict)
    self.allowed_action_types = allowed_action_types
    self.allowed_action_space = self.env.action_space.get_allowed_subdict(self.allowed_action_types)
    self.goal_space = goal_space
    self.name = name

    # TODO: the option itself needs to have a goal that may not be the same as the env's goal.


  # def take_action(self, observation):
  #   action_prob = self.policy_fn(observation)
  #   return action_prob

  def run_until_termination(self, observation, goal, is_eval=False):
    raise NotImplementedError('use api from getdata()')
    last_observation = observation
    ret = {
      'state': [],
      'action': [],
      'action_prob': [],
    }
    raw_rewards = []
    # TODO: add allowed action types and change that to suit the latest MutuallyExclusiveDict.
    should_terminate = False
    while not should_terminate:
      get_data_utils.replace_goal_in_observation(last_observation, goal)
      action, action_matrix, action_prob = policy_utils.get_actions(self.policy_fn, get_data_utils.add_batch_dim(last_observation), self.env.action_space, self.allowed_action_types, is_eval=is_eval)
      observation, reward, done, info = self.env.step(action)
      observation = get_data_utils.convert_env_observation(observation)
      raw_rewards.append(reward)

      ret['state'].append(last_observation)
      ret['action'].append(action_matrix)
      ret['action_prob'].append(action_prob)
      last_observation = observation

      should_terminate = done or self.termination_fn(observation)
    ret['rewards'] = env_rl_utils.compute_discounted_cumulative_reward(raw_rewards)
    return ret

  def get_action_space(self):
    return self.allowed_action_space
