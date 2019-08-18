"""Contains logic describing the causal relation between 'up' key and going forward."""


class GoStraightCausalModel(object):
  """
  When pressing the up key, if there is nothing blocking the agent, the agent's position will change by (dx,dy)
  which is decided by its facing direction.

  Allowed states: any
  Allowed action_input: up
  Goal states: any \in (x+k*dx, y+k*dy)


  """

  def __init__(self):
    pass

  def get_allowed_actions(self):
    pass

  def generate_episode(self):
    """Returns (s, g, [a1,a2,...], [s1, s2, ...])"""
    pass

  def do_one_step_reasoning(self, s, a):
    """Returns the next state given the current state and action_input."""
    pass


class GoStraightWorkerHardcoded(object):
  """Mimics a well-trained NN"""
  def __init__(self, env):
    self.env = env
    pass

  def set_goal(self, goal):
    self.goal = goal

  def step(self, s):
    if s + self.env.dir_vec() == self.goal:
      action = self.env.actions.forward
      done = True
      return action, done
    else:
      action = self.env.actions.forward
      done = False
      return action, done

