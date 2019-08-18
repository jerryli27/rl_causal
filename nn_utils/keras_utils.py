from tensorflow.python.keras.api._v2 import keras


def identity_loss(y_true, y_pred):
  # Maximize.
  return keras.backend.mean(y_pred)


def neg_identity_loss(y_true, y_pred):
  # Maximize.
  return - keras.backend.mean(y_pred)