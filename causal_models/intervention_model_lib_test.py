from causal_models import intervention_model_lib

import numpy as np
import tensorflow as tf

class TestInterventionModel(tf.test.TestCase):
  def test_get_intervention_model(self):
    batch, s_types_dim, s_embed_dim = 1, 1, 10
    _, a_types_dim, a_embed_dim = 1, 2, 10

    state_embed = tf.random.normal((batch, s_types_dim, s_embed_dim))

    # action_embed = tf.zeros((batch, a_types_dim, a_embed_dim))
    action = tf.zeros((batch, a_types_dim), dtype=tf.int64)
    action_embed = tf.one_hot(action, a_embed_dim, name='action_one_hot')

    causal_relation = tf.ones((s_types_dim, a_types_dim))

    model = intervention_model_lib.InterventionModel()
    intervened_state_embed = model((state_embed, action_embed, causal_relation))
    self.assertNotEmpty(intervened_state_embed)
    print(intervened_state_embed)


if __name__ == '__main__':
  tf.test.main()