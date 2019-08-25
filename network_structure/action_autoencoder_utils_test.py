from network_structure import autoencoder_utils

import gym
import numpy as np
import tensorflow as tf
from env_utils.spaces import me_dict_utils

def get_actions(action_space, batch):
  num_action_types = len(action_space.spaces)
  action_type_input = tf.constant([0 for _ in range(batch)], dtype=tf.int64)
  action_type_one_hot = tf.one_hot(action_type_input, num_action_types, name='action_type_one_hot')

  native_a_types_dim, native_a_embed_dim = 1, 5
  native_action = tf.zeros((batch, native_a_types_dim,), dtype=tf.int64)
  native_action_one_hot = tf.one_hot(native_action, native_a_embed_dim, name='native_action_one_hot')

  option_1_a_types_dim, option_1_a_embed_dim = 1, 7
  option_1_action = tf.zeros((batch, option_1_a_types_dim,), dtype=tf.int64)
  option_1_action_one_hot = tf.one_hot(option_1_action, option_1_a_embed_dim, name='option_1_action_one_hot')
  action_vec_inputs = [native_action_one_hot, option_1_action_one_hot]
  return action_type_one_hot, action_vec_inputs

class TestActionAutoencoder(tf.test.TestCase):
  def test_encoder(self):
    action_space = me_dict_utils.MutuallyExclusiveDict({
      'native': gym.spaces.MultiDiscrete([5,]),
      'option_1':  gym.spaces.MultiDiscrete([7,]),
    })
    num_action_types = len(action_space.spaces)
    batch = 2
    action_type_one_hot, action_vec_inputs = get_actions(action_space, batch)
    embed_dim = 10

    model = autoencoder_utils.MutuallyExclusiveActionEncoder(action_space, embed_dim)
    action_embed = model((action_type_one_hot, *action_vec_inputs))
    self.assertNotEmpty(action_embed)
    self.assertEqual((batch, num_action_types, embed_dim), action_embed.shape)


  def test_decoder(self):
    action_space = me_dict_utils.MutuallyExclusiveDict({
      'native': gym.spaces.MultiDiscrete([5,]),
      'option_1':  gym.spaces.MultiDiscrete([7,]),
    })
    num_action_types = len(action_space.spaces)
    batch = 2
    action_type_one_hot, action_vec_inputs = get_actions(action_space, batch)
    embed_dim = 10

    model = autoencoder_utils.MutuallyExclusiveActionEncoder(action_space, embed_dim)
    action_embed = model((action_type_one_hot, *action_vec_inputs))

    decoder_model = autoencoder_utils.MutuallyExclusiveActionDecoder(action_space, embed_dim)
    predicted_action_vecs = decoder_model((action_type_one_hot, action_embed))
    self.assertNotEmpty(predicted_action_vecs)
    for i in range(num_action_types):
      self.assertEqual(action_vec_inputs[i].shape, predicted_action_vecs[i].shape)

if __name__ == '__main__':
  tf.test.main()