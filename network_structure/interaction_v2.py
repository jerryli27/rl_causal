"""See DIIN. https://github.com/YerevaNN/DIIN-in-Keras/blob/master/layers/interaction.py"""

from tensorflow import keras

K = keras.backend


class Interaction(keras.layers.Layer):

  def build(self, input_shape):
    assert isinstance(input_shape, list)
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
                                  shape=(input_shape[0][1],input_shape[1][1],input_shape[0][2],input_shape[1][2],),
                                  initializer='lecun_normal',
                                  regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                                  trainable=True)
    super(Interaction, self).build(input_shape)


  def call(self, inputs, **kwargs):
    assert len(inputs) == 2
    premise_encoding, hypothesis_encoding = inputs


    # (3x1xmx1) X (3xmxnx2) X (1xnx1x2)
    # -> [(3x1xmxdup2) X (3xmxnx2)] X (1xnx1x2)
    # -> (3x1xnx2) X (1xnx1x2)
    # -> [(3x1xnx2) X (dup3xnx1x2)]
    # -> (3x1x1x2)->3x2

    # Perform element-wise multiplication for each row of premise and hypothesis
    # For every i, j premise_row[i] * hypothesis_row[j]
    # betta(premise_encoding[i], hypothesis_encoding[j]) = premise_encoding[i] * hypothesis_encoding[j]

    # => we can do the following:
    # 1. broadcast premise    to shape (batch, p, h, d)
    # 2. broadcast hypothesis to shape (batch, p, h, d)
    # perform premise * hypothesis

    # In keras this operation is equivalent to reshaping premise (batch, p, 1, d), hypothesis (batch, 1, h, d)
    # And then compute premise * hypothesis
    # (3xdup2x1xm)
    premise_encoding = keras.backend.expand_dims(keras.backend.expand_dims(premise_encoding, axis=2), axis=3)  # (batch, p, 1, 1, d_p)
    hypothesis_encoding = keras.backend.expand_dims(keras.backend.expand_dims(hypothesis_encoding, axis=1), axis=4)  # (batch, 1, h, d_h, 1)

    # premise_encoding = keras.backend.repeat_elements(keras.backend.repeat_elements(premise_encoding, rep=, axis=2), axis=3)

    kernal_w_batch = keras.backend.expand_dims(self.kernel, axis=0)

    dot_1 = keras.backend.batch_dot(premise_encoding, kernal_w_batch, axes=[4,3])
    dot_2 = keras.backend.batch_dot(dot_1, hypothesis_encoding, axes=[4,3])
    # (3x2x1x1)->3x2
    ret = keras.backend.squeeze(keras.backend.squeeze(dot_2, axis=4), axis=3)

    # Compute interaction tensor I = betta(premise, hypothesis)
    # I = premise_encoding * hypothesis_encoding
    return ret

  def compute_output_shape(self, input_shape):
    premise_shape = input_shape[0]
    hypothesis_shape = input_shape[1]

    # (batch, p, d), (batch, h, d) => (batch, p, h, d)
    assert len(premise_shape) == len(hypothesis_shape) == 3
    assert premise_shape[0] == hypothesis_shape[0]
    # assert premise_shape[2] == hypothesis_shape[2]
    batch = premise_shape[0]
    p = premise_shape[1]
    h = hypothesis_shape[1]
    # d = hypothesis_shape[2]

    return batch, p, h
