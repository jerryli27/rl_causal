#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import time
from optparse import OptionParser

import custom_envs
import gym


def main():
  parser = OptionParser()
  parser.add_option(
    "-e",
    "--env-name",
    dest="env_name",
    help="gym environment to load",
    default='TwoDigits-v0'
  )
  (options, args) = parser.parse_args()

  # Load the gym environment
  env = gym.make(options.env_name)

  def resetEnv():
    env.reset()
    if hasattr(env, 'goal_input'):
      print('goal_input: %s' % env.goal)

  resetEnv()

  # Create a window to render into
  renderer = env.render('human')

  def take_action(action):
    if action == 'BACKSPACE':
      resetEnv()
      return

    if action == 'ESCAPE':
      sys.exit(0)

    action = int(action)
    obs, reward, done, info = env.step(action)

    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
      print('done!')
      resetEnv()


  while True:
    env.render('human')
    time.sleep(0.01)

    curr_action = input('Action?')
    take_action(curr_action)


if __name__ == "__main__":
  main()
