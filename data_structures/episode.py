def backfill_discounted_cumulative_reward(step_results, gamma):
  dcr = [0.0 for _ in range(len(step_results))]
  dcr[-1] = step_results[-1].reward
  for i in range(len(step_results) - 2, -1, -1):
    dcr[i] = dcr[i + 1] * gamma + step_results[i].reward
  for i in range(len(step_results)):
    step_results[i].cumulative_reward = dcr[i]
  return

class Episode(object):
  def __init__(self, init_state, step_results, gamma):
    self.init_state = init_state
    self.step_results = step_results
    backfill_discounted_cumulative_reward(self.step_results, gamma=gamma)
