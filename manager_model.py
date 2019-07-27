import numpy as np

import critic

class ManagerQLearning(object):

  def __init__(self, workers, heuristic_critic, env, hparams):
    self.workers = workers
    self.heuristic_critic = heuristic_critic
    self.trained_critic = critic.Critic()
    self.env = env
    self.hparams = hparams

    self.curr_worker = None

  def step(self, s):
    # (optional) do mcts to pick the best action.
    # The action space for the manager is (worker_id, goal)

    if self.curr_worker is None:
      allowed_states = self.env.get_state_space()
      qs = []
      for worker_id in range(len(self.workers)):
        q_per_goal = []
        for goal in allowed_states:
          q = (self.hparams.critic_alpha * self.heuristic_critic.get_value(s, goal) +
               (1 - self.hparams.critic_alpha) * self.trained_critic.get_value(s, goal, worker_id))
          q_per_goal.append(q  ;[])
        qs.append(q_per_goal)

      # Do epsilon-explore.
      max_q_index = np.argmax(qs)
      worker, goal = self.workers[max_q_index[0]], allowed_states[max_q_index[1]]
      worker.set_goal(goal)
      self.curr_worker = worker

    return self.post_process_step(self.curr_worker.step(s))

  def post_process_step(self, step_ret):
    # Record the trajectory. If the current episode ended, update the critic and the policy.
    action, done = step_ret
    if done:
      self.curr_worker = None
    return action

  def update(self, trajectory):
    self.trained_critic.update(trajectory)