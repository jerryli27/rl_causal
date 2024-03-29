import functools
import gin
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

from env_utils.spaces import me_dict_utils

def get_leaky_relu():
  return functools.partial(keras.activations.relu, alpha=0.1)


@gin.configurable
def get_action_encoder_model(action_space, hidden_dim=64, num_hidden=2, embed_dim=16):
  if isinstance(action_space, gym.spaces.MultiDiscrete) or isinstance(action_space, gym.spaces.Box):
    # For now, let all different type of native actions share the same embedding...
    layers = [
      keras.layers.Flatten(),
    ]
    for i in range(num_hidden):
      layers.append(keras.layers.Dense(hidden_dim,
                                       activation=get_leaky_relu(),
                                       kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                       bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                       name='hidden_%d' %i))
    layers.append(keras.layers.Dense(embed_dim,
                                     kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                     bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                     name='embed'),)
  else:
    raise NotImplementedError

  model = keras.models.Sequential(layers, name='encoder')
  return model


def get_action_decoder_model(action_space, hidden_dim=64, num_hidden=2, name='decoder'):
  if isinstance(action_space, gym.spaces.MultiDiscrete):
    # Defaults to one-hot encoding.
    num_types = action_space.nvec.shape[0]
    max_num_classes = np.max(action_space.nvec)
    layers = []
    for i in range(num_hidden):
      layers.append(keras.layers.Dense(hidden_dim,
                                       activation=get_leaky_relu(),
                                       kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                       bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                       name='hidden_%d' %i))
    layers.extend([
      keras.layers.Dense(num_types * max_num_classes,
                         kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                         bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                         name='logits_fc'),
      keras.layers.Reshape((num_types, max_num_classes), name='logits'),
      keras.layers.Softmax(name='prob'),
    ])
  elif isinstance(action_space, gym.spaces.Box):
    # Defaults to one-hot encoding.
    output_shape = action_space.shape
    output_dim = np.prod(output_shape)
    layers = []
    for i in range(num_hidden):
      layers.append(keras.layers.Dense(hidden_dim,
                                       activation=get_leaky_relu(),
                                       kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                       bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                       name='hidden_%d' %i))
    layers.extend([
      keras.layers.Dense(output_dim,
                         kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                         bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                         name='pred_fc'),
      keras.layers.Reshape(output_shape, name='pred'),
    ])
  else:
    raise NotImplementedError
  model = keras.models.Sequential(layers=layers, name=name)
  return model

class MutuallyExclusiveActionBaseLayer(keras.layers.Layer):
  def __init__(self, space, embed_dim, **kwargs):
    super().__init__(**kwargs)
    self.space = space
    self.embed_dim = embed_dim
    self.num_types = len(space.spaces)
    self.types_to_i = {k: i for i, k in enumerate(space.spaces.keys())}
    self.types_i_to_name = {i: k for i, k in enumerate(space.spaces.keys())}

  def build(self, input_shape):
    pass


class MutuallyExclusiveActionEncoder(MutuallyExclusiveActionBaseLayer):
  def __init__(self, action_space, embed_dim, **kwargs):
    super().__init__(action_space, embed_dim, **kwargs)
    if not isinstance(action_space, me_dict_utils.MutuallyExclusiveDict):
      raise ValueError('Can only accept MutuallyExclusiveDict action space.')
    self.action_encoders = {
      i: get_action_encoder_model(self.space.spaces[k], embed_dim=self.embed_dim, )
      for i, k in enumerate(action_space.spaces.keys())
    }

  def call(self, inputs, **kwargs):
    action_type_one_hot = inputs[0]
    action_vec_inputs = inputs[1:]
    action_embed = tf.TensorArray(tf.float32, size=self.num_types)
    for i in range(self.num_types):
      action_vec = action_vec_inputs[i]
      action_embed_slice = self.action_encoders[i](action_vec)
      action_embed_mask = keras.backend.expand_dims(action_type_one_hot[:, i], axis=-1)
      action_embed_slice = action_embed_slice * action_embed_mask
      action_embed = action_embed.write(i, action_embed_slice)
    action_embed = action_embed.stack()  # Stacked on the first dim
    action_embed = keras.backend.permute_dimensions(action_embed, pattern=(1, 0, 2))
    action_embed = keras.backend.reshape(action_embed, (-1, self.num_types, self.embed_dim))
    return action_embed

  def compute_output_shape(self, input_shape):
    action_type_one_hot = input_shape[0]
    return (action_type_one_hot[0], self.num_types, self.embed_dim)


class MutuallyExclusiveActionDecoder(MutuallyExclusiveActionBaseLayer):
  def __init__(self, action_space, embed_dim, **kwargs):
    super().__init__(action_space, embed_dim, **kwargs)
    if not isinstance(action_space, me_dict_utils.MutuallyExclusiveDict):
      raise ValueError('Can only accept MutuallyExclusiveDict action space.')
    self.action_decoders = {
      i: get_action_decoder_model(self.space.spaces[k], name=k)
      for i, k in enumerate(action_space.spaces.keys())
    }

  def call(self, inputs, **kwargs):
    action_embeds = inputs
    # action_type_one_hot = inputs[0]
    # action_embeds = inputs[1]
    action_vecs = []
    for i in range(self.num_types):
      action_embed = action_embeds[:, i]
      action_vec_slice = self.action_decoders[i](action_embed)
      action_vecs.append(action_vec_slice)
    return action_vecs


class DictSpaceEncoder(MutuallyExclusiveActionBaseLayer):
  def __init__(self, action_space, embed_dim, **kwargs):
    super().__init__(action_space, embed_dim, **kwargs)
    self.encoders = {
      i: get_action_encoder_model(self.space.spaces[k], embed_dim=self.embed_dim, )
      for i, k in enumerate(action_space.spaces.keys())
    }

  def call(self, inputs, **kwargs):
    embed = tf.TensorArray(tf.float32, size=self.num_types)
    for i in range(self.num_types):
      vec = inputs[i]
      embed_slice = self.encoders[i](vec)
      embed = embed.write(i, embed_slice)
    embed = embed.stack()  # Stacked on the first dim
    embed = keras.backend.permute_dimensions(embed, pattern=(1, 0, 2))
    embed = keras.backend.reshape(embed, (-1, self.num_types, self.embed_dim))
    return embed

  def compute_output_shape(self, input_shape):
    batch = input_shape[0]
    return (batch, self.num_types, self.embed_dim)


class DictSpaceDecoder(MutuallyExclusiveActionBaseLayer):
  def __init__(self, action_space, embed_dim, **kwargs):
    super().__init__(action_space, embed_dim, **kwargs)
    self.decoders = {
      i: get_action_decoder_model(self.space.spaces[k], name=k)
      for i, k in enumerate(action_space.spaces.keys())
    }

  def call(self, inputs, **kwargs):
    embeds = inputs
    action_vecs = []
    for i in range(self.num_types):
      action_embed = embeds[:, i]
      action_vec_slice = self.decoders[i](action_embed)
      action_vecs.append(action_vec_slice)
    return action_vecs


def get_mse_loss(action_vec, predicted_action_vec):
  loss = []
  for i in range(len(action_vec)):
    curr_action_loss = keras.backend.sum(keras.losses.mse(action_vec[i], predicted_action_vec[i]))
    # curr_action_loss = keras.losses.mse(action_vec[i], predicted_action_vec[i])
    loss.append(curr_action_loss)
  loss = keras.backend.sum(loss)
  return loss


def get_categorical_crossentropy_loss(action_vec, predicted_action_vec):
  loss = []
  for i in range(len(action_vec)):
    curr_action_loss = keras.backend.sum(keras.losses.categorical_crossentropy(action_vec[i], predicted_action_vec[i]))
    # curr_action_loss = keras.losses.mse(action_vec[i], predicted_action_vec[i])
    loss.append(curr_action_loss)
  loss = keras.backend.sum(loss)
  return loss


def get_action_type_from_action_embed(action_embed):
  assert len(action_embed.shape) == 2 or len(action_embed.shape) == 3
  return np.argmax(np.sum(action_embed, axis=-1), axis=-1)