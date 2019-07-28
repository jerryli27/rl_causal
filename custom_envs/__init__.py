import custom_envs.envs
from gym.envs.registration import registry, register, make, spec

# Algorithmic
# ----------------------------------------

register(
  id='TwoDigits-v0',
  entry_point='custom_envs.envs:TwoDigitsEnv',
  max_episode_steps=200,
  reward_threshold=25.0,
)