from tensorflow import keras

def get_weights_with_name(model):
  names = [weight.name for layer in model.layers for weight in layer.weights]
  weights = model.get_weights()

  weights_dict = {}
  for name, weight in zip(names, weights):
    weights_dict[name] = weight
  return weights_dict


def print_outputs(input_tensors, output_tensors, input_data, batch_size=8):
  model = keras.models.Model(inputs=input_tensors, outputs=output_tensors)
  y = model.predict(x=input_data, batch_size=batch_size)
  print(y)
  # action_vec_inputs_decoded_fn = keras.backend.function(inputs=action_input, outputs=[action_vec_inputs, action_vec_decoded])
  # # action_vec_inputs_decoded_output = action_vec_inputs_decoded_fn(x_test[action_input.name])
  # action_vec_inputs_decoded_output = action_vec_inputs_decoded_fn((x_test['action_type'], x_test['action_native']))
  # action_vec_inputs_decoded_output = action_vec_inputs_decoded_fn({
  #   'action_type': x_test['action_type'],
  #   'action_native': x_test['action_native']})
  # print(action_vec_inputs_decoded_output)