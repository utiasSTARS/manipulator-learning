import tensorflow as tf

from manipulator_learning.learning.agents.tf.common import Actor, CNNActor


class EnsembleActor(tf.keras.Model):
    def __init__(self, input_dim, action_dim, num_actors=5, img_dim=None, num_head_hidden=256):
        super().__init__()
        if img_dim is not None:
            self.actors = [CNNActor(img_dim, input_dim, action_dim, True, num_head_hidden) for _ in range(num_actors)]
        else:
            self.actors = [Actor(input_dim, action_dim, num_head_hidden) for _ in range(num_actors)]
        self.different_data_per_actor = False

    def call(self, inputs, force_same_inputs_all=False):
        # for training only, gets raw outputs of each actor
        if self.different_data_per_actor and not force_same_inputs_all:
            # give each input to a different actor
            outs = [actor(inputs[i]) for i, actor in enumerate(self.actors)]
        else:
            # assume input is just a regular batch, all actors get same data
            outs = [actor(inputs) for actor in self.actors]
        return tf.concat([outs], -1)

    def get_action(self, inputs):
        mean, _ = self.inference(inputs)
        return mean

    def inference(self, inputs):
        # get mean and diagonal var
        outs = self.call(inputs, force_same_inputs_all=True)
        mean, var = tf.nn.moments(outs, axes=0)
        return tf.squeeze(mean), tf.squeeze(var)

    def mean_with_doubt(self, inputs):
        mean, var = self.inference(inputs)
        return mean, tf.norm(var)
