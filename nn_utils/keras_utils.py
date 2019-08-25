from tensorflow.python.keras.api._v2 import keras


def identity_loss(y_true, y_pred):
  # Maximize.
  return keras.backend.mean(y_pred)


def neg_identity_loss(y_true, y_pred):
  # Maximize.
  return - keras.backend.mean(y_pred)


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def reparametrization(z_mean, z_log_var):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1:]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = keras.backend.random_normal(shape=(batch, *dim))
    ret = z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon
    ret = keras.layers.Reshape((*dim,))(ret)
    return ret