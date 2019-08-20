import gym
from tensorflow import keras

from env_utils.spaces import me_dict_utils


def get_action_encoder_model(action_space, embed_dim):
  if isinstance(action_space, gym.spaces.MultiDiscrete):
    # First turn each into one-hot.
    # Then encode the one-hot.
    # TODO.
  else:
    raise NotImplementedError


def get_action_decoder_model(action_space, embed_dim):
  if isinstance(action_space, gym.spaces.MultiDiscrete):
    # First turn each into one-hot.
    # Then encode the one-hot.
    # TODO.
  else:
    raise NotImplementedError


class MutuallyExclusiveActionEncoder(object):
  def __init__(self, action_space, embed_dim):
    if not isinstance(action_space, me_dict_utils.MutuallyExclusiveDict):
      raise ValueError('Can only accept MutuallyExclusiveDict action space.')
    self.action_space = action_space
    self.embed_dim = embed_dim

    self.num_action_types = len(action_space.spaces)
    self.action_types_to_i = {k: i for i, k in enumerate(action_space.spaces.keys())}
    self.action_encoders = {
      k: get_action_encoder_model(self.action_space.spaces[k], self.embed_dim)
      for k in action_space.spaces.keys()
    }
    self.action_decoders = {
      k: get_action_decoder_model(self.action_space.spaces[k], self.embed_dim)
      for k in action_space.spaces.keys()
    }


  def encode(self, action):
    if not self.action_space.contains(action):
      raise ValueError('action_space does not contain the given action.')
    action_type, action_val = action
    action_type_i = self.action_types_to_i[action_type]
    # action_space_for_type_i = self.action_space.spaces[action_type]
    action_encoder = self.action_encoders[action_type]
    action_embedding = action_encoder(action_val)
    return action_embedding