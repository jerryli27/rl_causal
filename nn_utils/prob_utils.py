from tensorflow import keras


def smooth_one_hot(one_hot_probs, label_smoothing=0):
  if label_smoothing > 0:
    num_classes = keras.backend.int_shape(one_hot_probs)[-1]
    smooth_positives = 1.0 - label_smoothing
    smooth_negatives = float(label_smoothing) / num_classes
    one_hot_probs = one_hot_probs * smooth_positives + smooth_negatives
  return one_hot_probs



def inverse_softmax(one_hot_probs, label_smoothing=0):
  """

  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes
  :param label_smoothing:
  :return:
  """
  one_hot_probs = smooth_one_hot(one_hot_probs, label_smoothing)
  logits = keras.backend.log(one_hot_probs)
  return logits
