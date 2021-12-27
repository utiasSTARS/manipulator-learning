"""Common stuff for implementations of actors. Originally based on
https://github.com/google-research/google-research/blob/master/dac/common.py """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

import manipulator_learning.learning.agents.tf.cnn_models as cnn_models


def convert_env_obs_to_tuple(obs, combine_rgb_depth=True):
    obs = copy.deepcopy(obs)
    for k in obs.keys():
        obs[k] = np.expand_dims(obs[k], 0)  # batch index
    depth = np.expand_dims(obs['depth'], -1)  # channel index
    img = tf.image.convert_image_dtype(obs['img'], tf.float32)

    if combine_rgb_depth:
        return (tf.concat([img, depth], axis=-1), obs['obs'].astype('float32'))
    else:
        return (img, depth, obs['obs'].astype('float32'))


class CNNActorNoDepthNoState(tf.keras.Model):
    def __init__(self, img_dim, action_dim, channels=(64, 32, 32), kernel_sizes=(7, 4, 3), strides=(4, 2, 1),
                 spatial_softmax=False, num_ensemble=1, num_head_hidden=256):
        super().__init__()
        self.conv = cnn_models.CNNBasic(img_dim, channels, kernel_sizes, strides, spatial_softmax_out=spatial_softmax,
                                        num_ensemble=num_ensemble)
        self.num_ensemble = num_ensemble
        if not spatial_softmax:
            self.conv_out_flat_shape = self.conv.conv_out_shape[0] * self.conv.conv_out_shape[1] * \
                                       self.conv.conv_out_shape[2]
        else:
            self.conv_out_flat_shape = self.conv.conv_out_shape[-1]

        # todo ensemble separation not currently maintained through head as done in CNNRGBDepthState
        self.head = Actor(int(self.conv_out_flat_shape / num_ensemble), action_dim, num_head_hidden,
                          num_ensemble, False)
        # self.head = Actor(self.conv_out_flat_shape, action_dim, num_head_hidden,
        #                   1, False)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, correct_input_size_for_ensemble=False):
        rgb = self.conv(inputs, correct_input_size_for_ensemble)
        rgb = self.flatten(rgb)
        rgb = tf.expand_dims(rgb, axis=0)
        out = self.head(rgb)
        out = tf.squeeze(out, axis=0)
        # convert to num_ensemble * batch_size * out_size
        out = tf.transpose(
            tf.reshape(out, [out.shape[0], self.num_ensemble, out.shape[1] // self.num_ensemble]), [1, 0, 2])
        return out


# class CNNActor(cnn_models.CNNRGBDepthState):
class CNNActor(cnn_models.CNNRGBDepthCombinedState):
    def __init__(self, img_dim, state_input_dim, action_dim, spatial_softmax=True, num_head_hidden=256,
                 num_ensemble=1):
        super().__init__(img_dim, spatial_softmax=spatial_softmax, num_ensemble=num_ensemble)

        self.state_input_dim = state_input_dim
        # taking care of ensemble input modification in CNNRGBDepthState call function
        input_dim_without_ensemble = int(self.conv_out_flat_shape / num_ensemble + state_input_dim)
        self.head = Actor(input_dim_without_ensemble, action_dim, num_head_hidden,
                          num_ensemble=num_ensemble, modify_inputs_for_ensemble=False)

    def get_action(self, inputs):
        mean, _ = self.inference(inputs)
        return mean

    def inference(self, inputs, also_output_ssam_values=False):
        # get mean and diagonal var
        if also_output_ssam_values:
            outs, ssam_out, raw_spatial_softmax = self.call(inputs, also_output_ssam_values=True)
        else:
            outs = self.call(inputs)
        mean, var = tf.nn.moments(outs, axes=0)
        if also_output_ssam_values:
            return tf.squeeze(mean), tf.squeeze(var), ssam_out, raw_spatial_softmax
        else:
            return tf.squeeze(mean), tf.squeeze(var)

    def mean_with_doubt(self, inputs):
        mean, var = self.inference(inputs)
        return mean, tf.norm(var)


class Actor(tf.keras.Model):
    """Implementation of a determistic policy."""

    def __init__(self, input_dim, action_dim, num_head_hidden=256, num_ensemble=1,
                 modify_inputs_for_ensemble=True):
        """Initializes a policy network.

        Args:
          input_dim: size of the input space
          action_dim: size of the action space
        """
        super(Actor, self).__init__()
        self.num_ensemble = num_ensemble
        self.modify_inputs_for_ensemble = modify_inputs_for_ensemble
        if num_ensemble > 1:
            self.main = tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    input_shape=(1, input_dim * num_ensemble),
                    filters=num_head_hidden * num_ensemble,
                    activation='relu',
                    kernel_size=1,
                    kernel_initializer=tf.keras.initializers.Orthogonal(),
                    groups=num_ensemble),
                tf.keras.layers.Conv1D(
                    filters=num_head_hidden * num_ensemble,
                    activation='relu',
                    kernel_size=1,
                    kernel_initializer=tf.keras.initializers.Orthogonal(),
                    groups=num_ensemble),
                tf.keras.layers.Conv1D(
                    filters=action_dim * num_ensemble,
                    activation='tanh',
                    kernel_size=1,
                    kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                    groups=num_ensemble),
            ])

        else:
            self.main = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    # units=400,
                    units=num_head_hidden,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.Orthogonal(),
                    input_shape=(input_dim,)),
                tf.keras.layers.Dense(
                    # units=300,
                    units=num_head_hidden,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.Orthogonal()),
                tf.keras.layers.Dense(
                    units=action_dim,
                    activation='tanh',
                    kernel_initializer=tf.keras.initializers.Orthogonal(0.01))
            ])

    def call(self, inputs):
        """Performs a forward pass given the inputs.

        Args:
          inputs: a batch of observations (tfe.Variable).

        Returns:
          Actions produced by a policy.
        """
        if self.num_ensemble > 1 and self.modify_inputs_for_ensemble:
            inputs = tf.tile(inputs, tf.constant([1, self.num_ensemble]))
        return self.main(inputs)

    def get_action(self, inputs):
        mean, _ = self.inference(inputs)
        return mean

    def inference(self, inputs):
        # get mean and diagonal var
        outs = self.call(inputs)
        mean, var = tf.nn.moments(outs, axes=0)
        return tf.squeeze(mean), tf.squeeze(var)

    def mean_with_doubt(self, inputs):
        mean, var = self.inference(inputs)
        return mean, tf.norm(var)


class StochasticActor(tf.keras.Model):
    """Implements stochastic-actor."""

    def __init__(self, input_dim, action_dim):
        super(StochasticActor, self).__init__()

        self.mu = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=64,
                activation='tanh',
                kernel_initializer=tf.keras.initializers.Orthogonal(),
                input_shape=(input_dim,)),
            tf.keras.layers.Dense(
                units=64,
                activation='tanh',
                kernel_initializer=tf.keras.initializers.Orthogonal()),
            tf.keras.layers.Dense(
                units=action_dim,
                activation=None,
                kernel_initializer=tf.keras.initializers.Orthogonal(0.01))
        ])

        # We exponentiate the logsig to get sig (hence we don't need softplus).
        self.logsig = tf.Variable(
            name='logsig',
            shape=[1, action_dim],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True)

    @property
    def variables(self):
        """Overrides the variables property of tf.keras.Model.

        Required to include variables defined through tf.get_variable().

        Returns:
          List of trainable variables.
        """
        mu_var = self.mu.variables
        sig_var = self.logsig
        return mu_var + [sig_var]

    def dist(self, mu, sig):
        return tfp.distributions.MultivariateNormalDiag(
            loc=mu,
            scale_diag=sig)

    def sample(self, mu, sig):
        return self.dist(mu, sig).sample()

    def call(self, inputs):
        """Returns action distribution, given a state."""
        act_mu = self.mu(inputs)
        act_sig = tf.exp(tf.tile(self.logsig, [tf.shape(act_mu)[0], 1]))
        tf.assert_equal(act_mu.shape, act_sig.shape)

        act_dist = self.dist(act_mu, act_sig)
        return act_dist


class Critic(tf.keras.Model):
    """Implementation of state-value function."""

    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.main = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=64,
                input_shape=(input_dim,),
                activation='tanh',
                kernel_initializer=tf.keras.initializers.Orthogonal()),
            tf.keras.layers.Dense(
                units=64,
                activation='tanh',
                kernel_initializer=tf.keras.initializers.Orthogonal()),
            tf.keras.layers.Dense(
                units=1,
                activation=None,
                kernel_initializer=tf.keras.initializers.Orthogonal())
        ])

    def call(self, inputs):
        return self.main(inputs)


class CriticDDPG(tf.keras.Model):
    """Implementation of a critic base network."""

    def __init__(self, input_dim):
        """Initializes a policy network.

        Args:
          input_dim: size of the input space
        """
        super(CriticDDPG, self).__init__()

        self.main = tf.keras.Sequential([
            tf.keras.layers.Dense(
                # units=400,
                units=256,
                input_shape=(input_dim,),
                activation='relu',
                kernel_initializer=tf.keras.initializers.Orthogonal()),
            tf.keras.layers.Dense(
                # units=300,
                units=256,
                activation='relu',
                kernel_initializer=tf.keras.initializers.Orthogonal()),
            tf.keras.layers.Dense(
                units=1, kernel_initializer=tf.keras.initializers.Orthogonal())
        ])

    def call(self, inputs, actions):
        """Performs a forward pass given the inputs.

        Args:
          inputs: a batch of observations (tfe.Variable).
          actions: a batch of action.

        Returns:
          Values of observations.
        """
        x = tf.concat([inputs, actions], -1)
        return self.main(x)
