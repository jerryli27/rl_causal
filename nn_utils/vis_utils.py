from tensorflow import keras

def get_weights_with_name(model):
  names = [weight.name for layer in model.layers for weight in layer.weights]
  weights = model.get_weights()

  weights_dict = {}
  for name, weight in zip(names, weights):
    weights_dict[name] = weight
  return weights_dict