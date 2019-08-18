"""Implements an interaction network between two sets of embeddings."""

from tensorflow import keras

K = keras.backend


class Interaction(keras.layers.Layer):
  """Given two tensors [batch, a_type, a_dim], [batch, b_type, b_dim], computes the interaction between them.

  Outputs [batch, a_type, b_type]
  """
  def build(self, input_shape):
    assert isinstance(input_shape, list)
    # Create a trainable weight variable for this layer.
    # [a_type, b_type, a_dim, b_dim]
    self.kernel = self.add_weight(name='kernel',
                                  shape=(input_shape[0][1], input_shape[1][1], input_shape[0][2], input_shape[1][2],),
                                  initializer='lecun_normal',
                                  regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                  trainable=True)
    super(Interaction, self).build(input_shape)

  def call(self, inputs, **kwargs):
    assert len(inputs) == 2, 'Please call with (a_embed, b_embed) as inputs.'
    a_embed, b_embed = inputs
    # (batch, a_type, 1, 1, a_dim)
    a_embed = keras.backend.expand_dims(keras.backend.expand_dims(a_embed, axis=2), axis=3)
    # (batch, 1, b_type, b_dim, 1)
    b_embed = keras.backend.expand_dims(keras.backend.expand_dims(b_embed, axis=1), axis=4)
    kernal_w_batch = keras.backend.expand_dims(self.kernel, axis=0)  # [1, a_type, b_type, a_dim, b_dim]

    # Dot the last two dims with broadcasting. dot([1, a_dim], [a_dim, b_dim]) = [1, b_dim]
    dot_1 = keras.backend.batch_dot(a_embed, kernal_w_batch, axes=[4, 3])  # [batch, a_type, b_type, 1, b_dim]
    # Dot the last two dims with broadcasting. dot([1, b_dim], [b_dim, 1]) = [1, 1]
    dot_2 = keras.backend.batch_dot(dot_1, b_embed, axes=[4, 3])  # [batch, a_type, b_type, 1, 1]
    ret = keras.backend.squeeze(keras.backend.squeeze(dot_2, axis=4), axis=3)  # [batch, a_type, b_type]
    return ret

  def compute_output_shape(self, input_shape):
    a_shape = input_shape[0]
    b_shape = input_shape[1]
    assert len(a_shape) == len(b_shape) == 3
    assert a_shape[0] == b_shape[0]
    batch = a_shape[0]
    a_type = a_shape[1]
    b_type = b_shape[1]
    return batch, a_type, b_type
