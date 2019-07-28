
from tensorflow import keras

def multi_discrete_to_onehot_concat(num_classes):
  """Converts a vector of ints representing a MultiDiscrete object to a concatenated onehot vector."""
  # keras.utils.to_categorical(current_state)
  raise NotImplementedError('This may actually bring more trouble downstream during prediction. Cannot use softmax etc.')