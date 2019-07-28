class StepResult(object):
  def __init__(self, action, obs, reward, done, info):
    self.action = action
    self.obs, self.reward, self.done, self.info = obs, reward, done, info
