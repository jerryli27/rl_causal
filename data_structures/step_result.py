class StepResult(object):
  def __init__(self, action, obs, reward, done, info, action_info=None, cumulative_reward=None):
    self.action = action
    self.obs, self.reward, self.done, self.info, self.action_info, self.cumulative_reward= obs, reward, done, info, action_info, cumulative_reward
