# Generic model for CNN
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
from timeit import default_timer as timer


# from tensorflow.contrib.layers import spatial_softmax


# had to add specific spatial softmax since removed from contrib
# https://github.com/google-research/tensor2robot/blob/master/layers/spatial_softmax.py#L19
# https://github.com/tensorflow/addons/issues/1364
# https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/layers/python/layers/layers.py
def get_spatial_softmax(features, spatial_gumbel_softmax=False):
    """Computes the spatial softmax of the input features.
      Args:
        features: A tensor of size [batch_size, num_rows, num_cols, num_features]
        spatial_gumbel_softmax: If set to True, samples locations stochastically
          rather than computing expected coordinates with respect to heatmap.
      Returns:
        A tuple of (expected_feature_points, softmax).
        expected_feature_points: A tensor of size
          [batch_size, num_features * 2]. These are the expected feature
          locations, i.e., the spatial softmax of feature_maps. The inner
          dimension is arranged as [x1, x2, x3 ... xN, y1, y2, y3, ... yN].
        softmax: A Tensor which is the softmax of the features.
          [batch_size, num_rows, num_cols, num_features].
      """
    _, num_rows, num_cols, num_features = features.get_shape().as_list()

    # Create tensors for x and y positions, respectively
    x_pos = np.empty([num_rows, num_cols], np.float32)
    y_pos = np.empty([num_rows, num_cols], np.float32)

    # Assign values to positions
    for i in range(num_rows):
        for j in range(num_cols):
            x_pos[i, j] = 2.0 * j / (num_cols - 1.0) - 1.0
            y_pos[i, j] = 2.0 * i / (num_rows - 1.0) - 1.0

    x_pos = tf.reshape(x_pos, [num_rows * num_cols])
    y_pos = tf.reshape(y_pos, [num_rows * num_cols])

    # We reorder the features (norm3) into the following order:
    # [batch_size, NUM_FEATURES, num_rows, num_cols]
    # This lets us merge the batch_size and num_features dimensions, in order
    # to compute spatial softmax as a single batch operation.
    features = tf.reshape(
        tf.transpose(features, [0, 3, 1, 2]), [-1, num_rows * num_cols])

    if spatial_gumbel_softmax:
        # Temperature is hard-coded for now, make this more flexible if results
        # are promising.
        dist = tfp.distributions.RelaxedOneHotCategorical(
            temperature=1.0, logits=features)
        softmax = dist.sample()
    else:
        softmax = tf.nn.softmax(features)
    # Element-wise multiplication
    x_output = tf.multiply(x_pos, softmax)
    y_output = tf.multiply(y_pos, softmax)
    # Sum per out_size x out_size
    x_output = tf.reduce_sum(x_output, [1], keepdims=True)
    y_output = tf.reduce_sum(y_output, [1], keepdims=True)
    # Concatenate x and y, and reshape.
    expected_feature_points = tf.reshape(
        tf.concat([x_output, y_output], 1), [-1, num_features * 2])
    softmax = tf.transpose(
        tf.reshape(softmax, [-1, num_features, num_rows,
                             num_cols]), [0, 2, 3, 1])
    return expected_feature_points, softmax


class CNNRGBDepthCombinedState(tf.keras.Model):
    def __init__(self, img_dim, channels=(64, 32, 32), kernel_sizes=(7, 4, 3), strides=(4, 2, 1), spatial_softmax=False,
                 num_ensemble=1):
        """ As opposed to CNNRGBDepthState, the RGB and Depth channels are merged into a 4-channel input, primarily
        with the goal of speeding up training and inference. """
        super().__init__()
        if spatial_softmax and len(channels) == 3:  # todo this shouldn't necessarily be hardcoded
            kernel_sizes = (7, 1, 3)
            strides = (2, 1, 1)

        self.num_ensemble = num_ensemble
        self.spatial_softmax = spatial_softmax

        img_dim = list(img_dim)
        img_dim[-1] += 1  # one more channel for depth
        self.cnn = CNNBasic(img_dim, channels, kernel_sizes, strides, resnet_first_layer=True,
                            num_ensemble=num_ensemble,
                            spatial_softmax_out=spatial_softmax)

        if self.num_ensemble > 1:
            cnn_size = self.cnn.conv_out_shape[-1] // self.num_ensemble
            self.ensemble_joined_slices = [slice(i * cnn_size, (i + 1) * cnn_size) for i in range(self.num_ensemble)]

        if not spatial_softmax:
            self.conv_out_flat_shape = self.cnn.conv_out_shape[0] * self.cnn.conv_out_shape[1] * \
                                       self.cnn.conv_out_shape[2]
        else:
            self.conv_out_flat_shape = self.cnn.conv_out_shape[-1]

        self.flatten = tf.keras.layers.Flatten()
        # should be overwritten by other classes
        self.head = None

    def call(self, inputs, correct_input_size_for_ensemble=False, also_output_ssam_values=False):
        rgb_depth_in, state_in = inputs
        b_size = rgb_depth_in.shape[0]
        if also_output_ssam_values:
            rgb, raw_spatial_softmax = self.cnn(rgb_depth_in, correct_input_size_for_ensemble,
                                                also_output_raw_softmax=also_output_ssam_values)
        else:
            rgb = self.cnn(rgb_depth_in, correct_input_size_for_ensemble)

        if self.num_ensemble > 1:
            loop_start = timer()
            if not correct_input_size_for_ensemble:
                state_in_along_ens = tf.tile(tf.expand_dims(state_in, -1), (1, 1, self.num_ensemble))
            else:
                state_in_along_ens = tf.reshape(state_in,
                                                [b_size, self.num_ensemble, state_in.shape[-1] // self.num_ensemble])
                state_in_along_ens = tf.transpose(state_in_along_ens, [0, 2, 1])  # now batch x state x num_ensemble

            if not self.spatial_softmax:
                cnn_along_ens = tf.reshape(
                    rgb, [b_size, rgb.shape[1], rgb.shape[2], self.num_ensemble, rgb.shape[-1] // self.num_ensemble])
                cnn_along_ens = tf.transpose(cnn_along_ens, [0, 1, 2, 4, 3])
                cnn_along_ens = tf.reshape(
                    cnn_along_ens, [b_size, cnn_along_ens.shape[1] * cnn_along_ens.shape[2] * cnn_along_ens.shape[3],
                                    self.num_ensemble])
            else:
                cnn_along_ens = tf.reshape(rgb, [b_size, self.num_ensemble, rgb.shape[-1] // self.num_ensemble])
                cnn_along_ens = tf.transpose(cnn_along_ens, [0, 2, 1])
            # dimension of cnn_along_ens is now batch x rgb_out_flat x num_ensemble

            flat_for_head = self.flatten(
                tf.transpose(tf.concat([cnn_along_ens, state_in_along_ens], axis=1), [0, 2, 1]))
            flat_for_head = tf.expand_dims(flat_for_head, axis=0)  # needed since head now uses Conv1D for ensemble

            # print("5. loop time: ", timer() - loop_start)
            head_start = timer()
            raw_out = tf.squeeze(self.head(flat_for_head), axis=0)  # shape is batch_size * (out_size * num_ensemble)
            # convert to num_ensemble * batch_size * out_size
            out = tf.transpose(tf.reshape(
                raw_out, [raw_out.shape[0], self.num_ensemble, raw_out.shape[1] // self.num_ensemble]), [1, 0, 2])
            # print("6. head time: ", timer() - head_start)

        else:
            head_start = timer()
            cnn_flat = self.flatten(rgb)
            flat_for_head = tf.concat([cnn_flat, state_in], -1)
            out = self.head(flat_for_head)
            # print("head time: ", timer() - head_start)

        if also_output_ssam_values:
            return out, rgb, raw_spatial_softmax  # features are rgb, raw_spatial softmax are actual 2D softmax values
        else:
            return out


class CNNRGBDepthState(tf.keras.Model):
    def __init__(self, img_dim, channels=(64, 32, 32), kernel_sizes=(7, 4, 3), strides=(4, 2, 1), spatial_softmax=False,
                 num_ensemble=1):
        super().__init__()

        if spatial_softmax and len(channels) == 3:  # todo this shouldn't necessarily be hardcoded
            kernel_sizes = (7, 1, 3)
            strides = (2, 1, 1)

        self.num_ensemble = num_ensemble
        self.spatial_softmax = spatial_softmax

        self.rgb_cnn = CNNBasic(img_dim, [channels[0]], [kernel_sizes[0]], [strides[0]], resnet_first_layer=True,
                                num_ensemble=num_ensemble)
        self.depth_cnn = CNNBasic((img_dim[0], img_dim[1], 1), [64], [kernel_sizes[0]], [strides[0]],
                                  num_ensemble=num_ensemble, resnet_first_layer=True)

        joined_in_shape = [self.rgb_cnn.conv_out_shape[0], self.rgb_cnn.conv_out_shape[1],
                           self.rgb_cnn.conv_out_shape[2] + self.depth_cnn.conv_out_shape[2]]

        if num_ensemble > 1:
            joined_in_shape[2] = joined_in_shape[
                                     2] // num_ensemble  # since we change shapes for everything with num_ensemble arg

        self.joined_cnn = CNNBasic(joined_in_shape, channels[1:], kernel_sizes[1:], strides[1:],
                                   spatial_softmax_out=spatial_softmax, modify_inputs_for_ensemble=False,
                                   num_ensemble=num_ensemble)
        if self.num_ensemble > 1:
            rgb_size = self.rgb_cnn.conv_out_shape[-1] // self.num_ensemble
            depth_size = self.depth_cnn.conv_out_shape[-1] // self.num_ensemble
            joined_size = self.joined_cnn.conv_out_shape[-1] // self.num_ensemble
            self.ensemble_rgb_slices = [slice(i * rgb_size, (i + 1) * rgb_size) for i in range(self.num_ensemble)]
            self.ensemble_depth_slices = [slice(i * depth_size, (i + 1) * depth_size) for i in range(self.num_ensemble)]
            self.ensemble_joined_slices = [slice(i * joined_size, (i + 1) * joined_size) for i in
                                           range(self.num_ensemble)]

        if not spatial_softmax:
            self.conv_out_flat_shape = self.joined_cnn.conv_out_shape[0] * self.joined_cnn.conv_out_shape[1] * \
                                       self.joined_cnn.conv_out_shape[2]
        else:
            self.conv_out_flat_shape = self.joined_cnn.conv_out_shape[-1]

        self.flatten = tf.keras.layers.Flatten()
        # should be overwritten by other classes
        self.head = None

    def call(self, inputs, correct_input_size_for_ensemble=False):
        # # since we can't change the call signature of call, instead make "correct_input_sizes_for_ensemble" be an
        # # optional argument as a list parameter
        # if len(inputs) == 3:
        #   rgb_in, depth_in, state_in = inputs
        #   correct_input_size_for_ensemble = False
        # elif len(inputs) == 4:
        #   rgb_in, depth_in, state_in, correct_input_size_for_ensemble = inputs
        # else:
        #   raise ValueError("inputs into CNNRGBDepthState model should be a tuple of rgb, depth, state, with optional"
        #                    "argument for correct_input_size_for_ensemble")
        call_start = timer()
        rgb_in, depth_in, state_in = inputs

        tic = timer()
        rgb = self.rgb_cnn(rgb_in, correct_input_size_for_ensemble)
        print("1. rgb time: ", timer() - tic)
        tic = timer()
        depth = self.depth_cnn(depth_in, correct_input_size_for_ensemble)
        # depth = self.depth_cnn(rgb_in[:, :, :, :1], correct_input_size_for_ensemble)
        print("2. depth time: ", timer() - tic)

        tic = timer()
        if self.num_ensemble > 1:
            rgb_slices = [rgb[:, :, :, sl] for sl in self.ensemble_rgb_slices]
            depth_slices = [depth[:, :, :, sl] for sl in self.ensemble_depth_slices]
            rgb_depth_slices = [tf.concat([r_sl, d_sl], -1) for r_sl, d_sl in zip(rgb_slices, depth_slices)]
            rgb_depth_joined = tf.concat(rgb_depth_slices, -1)
        else:
            rgb_depth_joined = tf.concat([rgb, depth], -1)
        print("3. prep joined time: ", timer() - tic)

        tic = timer()
        joined = self.joined_cnn(rgb_depth_joined)  # these are now spatial softmax outs
        print("4. joined time: ", timer() - tic)

        if self.num_ensemble > 1:
            loop_start = timer()
            if not correct_input_size_for_ensemble:
                state_in_repeat_list = [state_in] * self.num_ensemble
            else:
                state_in_repeat_list = tf.split(state_in, self.num_ensemble, axis=-1)

            joined_slices_flat = tf.split(joined, self.num_ensemble, axis=-1)
            if not self.spatial_softmax:  # if spatial softmax, already flattened along non-batch dimension
                joined_slices_flat = [self.flatten(sl) for sl in joined_slices_flat]

            joined_for_head_slices = [tf.concat([j_sl, s_sl], -1) for j_sl, s_sl in zip(joined_slices_flat,
                                                                                        state_in_repeat_list)]
            joined_for_head = tf.concat(joined_for_head_slices, -1)

            joined_for_head = tf.expand_dims(joined_for_head, axis=0)  # needed since head now uses Conv1D for ensemble

            print("5. loop time: ", timer() - loop_start)
            head_start = timer()
            raw_out = tf.squeeze(self.head(joined_for_head), axis=0)  # shape is batch_size * (out_size * num_ensemble)
            out = tf.transpose(tf.reshape(raw_out, [raw_out.shape[0], self.num_ensemble,
                                                    raw_out.shape[1] // self.num_ensemble]),
                               [1, 0, 2])  # converted to num_ensemble * batch_size * out_size
            print("6. head time: ", timer() - head_start)
        else:
            head_start = timer()
            joined_flat = self.flatten(joined)
            joined_for_head = tf.concat([joined_flat, state_in], -1)
            out = self.head(joined_for_head)
            print("head time: ", timer() - head_start)

        print("call time: ", timer() - call_start)
        return out


class CNNBasic(tf.keras.Model):
    def __init__(self, in_shape, channels, kernel_sizes, strides, paddings=None,
                 activation='relu', use_maxpool=False, resnet_first_layer=False,
                 spatial_softmax_out=False, num_ensemble=1, modify_inputs_for_ensemble=True):
        super().__init__()
        self.spatial_softmax_out = spatial_softmax_out

        if paddings is None:
            paddings = ['valid' for _ in range(len(channels))]

        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        self.activation = activation  # relu, tanh, or sigmoid
        kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
        conv_layers = []
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones

        # h, w, in_channels = in_shape
        # in_channels = [in_channels] + list(channels[:-1])
        # modifications for ensembles based on grouped convolutions -- requires tensorflow>=2.3!!
        # see https://stackoverflow.com/questions/58374980/run-multiple-models-of-an-ensemble-in-parallel-with-pytorch
        # and https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
        self.num_ensemble = num_ensemble
        self.modify_inputs_for_ensemble = modify_inputs_for_ensemble
        channels = list(np.array(channels) * num_ensemble)
        in_shape = list(in_shape)
        in_shape[2] = in_shape[2] * num_ensemble

        conv_layers.extend(
            [tf.keras.layers.Conv2D(input_shape=in_shape, filters=channels[0], kernel_size=kernel_sizes[0],
                                    strides=strides[0], padding=paddings[0], activation=self.activation,
                                    kernel_initializer=kernel_init, groups=num_ensemble)])
        conv_layers.extend([tf.keras.layers.Conv2D(filters=c, kernel_size=k,
                                                   strides=s, padding=p, activation=self.activation,
                                                   kernel_initializer=kernel_init, groups=num_ensemble) for (c, k, s, p)
                            in
                            zip(channels[1:], kernel_sizes[1:], strides[1:], paddings[1:])])
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer])
            if maxp_stride > 1:
                sequence.append(tf.layers.MaxPooling2D(maxp_stride, maxp_stride))  # No padding

        if len(sequence) > 1:
            self.conv = tf.keras.Sequential([*sequence])
        else:
            self.conv = sequence[0]

        # set up simple FC head, should be overwritten
        self.head = None

        if spatial_softmax_out:
            # self.spatial_softmax = tf.keras.layers.Lambda(lambda x: spatial_softmax(x, temperature=1.0))
            self.conv_out_shape = [channels[-1] * 2]
        else:
            # surely there's a better way to do this, but oh well
            ex_in = np.ones([1, *in_shape], dtype='float32')
            self.conv_out_shape = self.conv(ex_in).shape[1:]

        # initialize first cnn layer using resnet101 weights,
        # first layer must be nn.Conv2d(3, 64, kernel_size=7)
        if resnet_first_layer:
            file_path = os.path.dirname(os.path.realpath(__file__))
            weights = np.load(file_path + '/assets/cafferesnet_layer1_weights.npy')
            biases = np.zeros(channels[0])
            weights = [weights.transpose([2, 3, 1, 0]), biases]

            if num_ensemble > 1:
                weights[0] = tf.tile(weights[0], tf.constant([1, 1, 1, num_ensemble]))

            if type(self.conv) == tf.keras.Sequential and len(self.conv.layers) > 1:
                first_conv = self.conv.layers[0]
            else:
                first_conv = self.conv

            rgb_filt_size = first_conv.weights[0].shape[2]
            if rgb_filt_size == 1:
                # adjustment for depth/b&w images
                weights[0] = np.expand_dims(weights[0][:, :, 0, :] * 3, 2)
            elif rgb_filt_size == 4:
                # adjustment for combined rgb + depth images
                weights[0] = tf.concat([weights[0] * .75, tf.expand_dims(weights[0][:, :, 0, :], 2) * .25], axis=2)
            first_conv.set_weights(weights)

            # else:
            #   rgb_filt_size = self.conv.weights[0].shape[2]
            #   if rgb_filt_size == 1:
            #     # adjustment for depth/b&w images
            #     weights[0] = np.expand_dims(weights[0][:, :, 0, :] * 3, 2)
            #   elif rgb_filt_size == 4:
            #   self.conv.set_weights(weights)

    def call(self, inputs, correct_input_size_for_ensemble=False, also_output_raw_softmax=False):
        if self.num_ensemble > 1 and self.modify_inputs_for_ensemble and not correct_input_size_for_ensemble:
            # input is B * H * W * C, want B * H * W * (C * num_ensemble)
            inputs = tf.tile(inputs, tf.constant([1, 1, 1, self.num_ensemble]))
        if self.spatial_softmax_out:
            conv_out = self.conv(inputs)
            # return self.spatial_softmax(conv_out)
            if also_output_raw_softmax:
                spatial_softmax, raw_softmax = get_spatial_softmax(conv_out)
                return spatial_softmax, raw_softmax
            else:
                spatial_softmax, _ = get_spatial_softmax(conv_out)
                return spatial_softmax
        return self.conv(inputs)
