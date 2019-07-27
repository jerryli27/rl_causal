from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class Empty1DEnv(MiniGridEnv):
  """
  Empty grid environment, no obstacles, sparse reward
  """

  def __init__(
      self,
      size=8,
      agent_start_pos=(1, 1),
      agent_start_dir=0,
  ):
    self.agent_start_pos = agent_start_pos
    self.agent_start_dir = agent_start_dir

    super().__init__(
      height=3,
      width=size,
      max_steps=4 * size,
      # Set this to True for maximum speed
      see_through_walls=True
    )

  def _gen_grid(self, width, height):
    # Create an empty grid
    self.grid = Grid(width, height)

    # Generate the surrounding walls
    self.grid.wall_rect(0, 0, width, height)

    # Place a goal square in the bottom-right corner
    self.grid.set(width - 2, height - 2, Goal())

    # Place the agent
    if self.agent_start_pos is not None:
      self.agent_pos = self.agent_start_pos
      self.agent_dir = self.agent_start_dir
    else:
      self.place_agent()

    self.mission = "get to the green goal square"


class Empty1DEnv5(Empty1DEnv):
  def __init__(self):
    super().__init__(size=5)


class EmptyRandom1DEnv5(Empty1DEnv):
  def __init__(self):
    super().__init__(size=5, agent_start_pos=None)


register(
  id='MiniGrid-Empty1D-5-v0',
  entry_point='custom_envs.envs:Empty1DEnv5'
)
register(
  id='MiniGrid-Empty1D-5-rand-v0',
  entry_point='custom_envs.envs:EmptyRandom1DEnv5'
)
