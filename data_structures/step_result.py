class StepResult(object):
  def __init__(self, action, obs, reward, done, info, action_info=None):
    self.action = action
    self.obs, self.reward, self.done, self.info, self.action_info = obs, reward, done, info, action_info
