import gym
from gym import error, logger, spaces, utils
from gym.utils import seeding
import numpy as np


class TwoDigitsEnv(gym.GoalEnv):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    # self.action_space = spaces.Discrete(5)
    self.action_space = spaces.MultiDiscrete([5,])

    self._hidden_state_space = spaces.MultiDiscrete([10, 2])
    # For this simple env, we assume that there is no hidden variables.
    # In the next version, it may be good to have a function that goes from
    # hidden state to observed state.
    self._observed_state_space = self._hidden_state_space
    self._reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=tuple())
    self.observation_space = gym.spaces.Dict({
      'observation': self._observed_state_space,
      'achieved_goal': self._observed_state_space,
      'desired_goal': self._observed_state_space,
      'reward': self._reward_space,
    })
    self.max_num_steps = 20

    # self.np_random = None
    self.seed()
    self.viewer = None
    self.state = None
    self.goal = None
    self.step_count = None

    self.steps_beyond_done = None

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    state = self.state

    # TODO: rename...It's not really delta.
    action_0 = action[0]
    next_state = np.copy(state)
    if action_0 != 0:
      next_state[0] = action_0
    else:
      # The last action_input is do nothing.
      pass

    # After intervention, the state evolves into the next following the transition prob.
    next_state[0] = (next_state[0] + 1) % 10
    # next_second_digit = next_first_digit % 2
    next_state[1] = (next_state[1] + 1) % 2
    assert self._observed_state_space.contains(next_state), 'internal error. Illegal next state'

    self.state = next_state
    self.step_count += 1
    if self.step_count >= self.max_num_steps:
      if self.steps_beyond_done is None:
        self.steps_beyond_done = 0
      else:
        if self.steps_beyond_done == 0:
          logger.warn(
            'You are calling \'step()\' even though this environment has already returned done = True. You should '
            'always call \'reset()\' once you receive \'done = True\' -- any further steps are undefined behavior.')
        self.steps_beyond_done += 1
    done = self._get_is_done()

    info = {'done': done, 'steps_beyond_done': self.steps_beyond_done}
    reward = self.compute_reward(self._get_achieved_goal(), self._get_desired_goal(), done)

    return self._get_observation(), reward, done, info

  def get_noop_action(self):
    return 0

  def _get_achieved_goal(self):
    return self.state

  def _get_desired_goal(self):
    return self.goal

  def _get_is_done(self):
    done = ((self.steps_beyond_done is not None) or
            self._is_desired_goal_achieved(self._get_achieved_goal(), self._get_desired_goal()))
    return done


  def _get_observation(self):
    """Returns the observation from the current state and goal_input."""
    ret = {
      'observation': self.state,
      'achieved_goal': self._get_achieved_goal(),
      'desired_goal': self._get_desired_goal(),
    }
    ret.update({'reward': self.compute_reward(ret['achieved_goal'], ret['desired_goal'])})
    return ret

  @staticmethod
  def _is_desired_goal_achieved(achieved_goal, desired_goal):
    return np.all(achieved_goal == desired_goal)

  def compute_reward(self, achieved_goal=None, desired_goal=None, info=None):
    """This reward ignores the steps boundary."""
    if achieved_goal is None:
      achieved_goal = self._get_achieved_goal()
    if desired_goal is None:
      desired_goal = self._get_desired_goal()
    done = info
    if done is None:
      done = self._get_is_done()
    if self._is_desired_goal_achieved(achieved_goal, desired_goal):
      return 1.0
    if done:
      return -1.0
    return 0.0

  def _get_random_goal(self):
    # For now, return a random goal_input. In the future, we may want to use a randomly generated
    # sequence of actions and compete our agent against that sequence for efficiency.
    return self._observed_state_space.sample()

  def reset(self):
    super(TwoDigitsEnv, self).reset()
    self.state = np.array([0, 0])
    self.goal = self._get_random_goal()
    self.step_count = 0
    self.steps_beyond_done = None
    return self._get_observation()

  def render(self, mode='human', close=False):
    print(self._get_observation())