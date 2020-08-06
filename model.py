from __future__ import division, print_function

__author__ = "Lauri Juvela, lauri.juvela@aalto.fi"

import numpy as np
import tensorflow as tf

_FLOATX = tf.float32

def get_weight_variable(name, shape=None, use_spectral_norm=False, update_u=True, initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    if shape is None:
        return tf.get_variable(name)
    else:
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)


def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX)):
    if shape is None:
        return tf.get_variable(name)
    else:
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)

class WaveNet():

    def __init__(self,
                 name,
                 input_channels=40,
                 output_channels=1,
                 residual_channels=64,
                 postnet_channels=256,
                 filter_width=3,
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 cond_channels=None,
                 cond_embed_dim=64,
                 use_dropout=False,
                 dropout_keep_prob=0.95,
                 mask_center_input=False):

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.postnet_channels = postnet_channels
        self.use_dropout = use_dropout
        self.dropout_keep_prob = dropout_keep_prob
        self.mask_center_input = mask_center_input

        if cond_channels is not None:
            self._use_cond = True
            self.cond_embed_dim = cond_embed_dim
            self.cond_channels = cond_channels
        else:
            self._use_cond = False

        self._name = name
        #self._create_variables()

    def get_receptive_field(self):
        # returns one sided receptive field of the model
        # symmetric receptive field is actually 2 * get_receptive_field() - 1

        # from dilated stack
        receptive_field = sum(self.dilations) * ((self.filter_width - 1) // 2)
        # from input layer
        receptive_field += (self.filter_width - 1) // 2 - 1
        # current sample
        receptive_field += 1

        return receptive_field

    def get_variable_list(self):
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
        if len(theta) == 0:
            raise RuntimeError("Model '{}' variabiles are not allocated until the first call of forward_pass".format(self._name))         
        return theta

    def _input_layer(self, main_input):
        with tf.variable_scope('input_layer'):

            fw = self.filter_width
            r = self.residual_channels

            W = get_weight_variable('W', (fw, self.input_channels, r))
            b = get_bias_variable('b', (r))

            if self.mask_center_input:
                assert fw % 2 == 1, "Filter width must be odd if masking the center value"
                ones = tf.ones(
                    shape=(fw//2, self.input_channels, r), dtype=W.dtype)
                zeros = tf.zeros(
                    shape=(1, self.input_channels, r), dtype=W.dtype)
                M = tf.concat([ones, zeros, ones], axis=0)
                W = M*W

            X = main_input

            Y = tf.nn.convolution(X, W, padding='SAME')
            Y += b

            if self.use_dropout:
                Y = tf.nn.dropout(Y, keep_prob=self.dropout_keep_prob)

            Y = tf.tanh(Y)

        return Y

    def _embed_cond(self, cond_input):
        with tf.variable_scope('embed_cond'):
            W = get_weight_variable(
                'W', (1, self.cond_channels, self.cond_embed_dim))
            b = get_bias_variable('b', (self.cond_embed_dim))

            Y = tf.nn.convolution(
                cond_input, W, padding='SAME')  # 1x1 convolution
            Y += b

            return tf.tanh(Y)

    def _conv_module(self, main_input, residual_input, module_idx, dilation, cond_input=None):
        with tf.variable_scope('conv_modules'):
            with tf.variable_scope('module{}'.format(module_idx)):

                fw = self.filter_width
                r = self.residual_channels
                s = self.postnet_channels

                X = main_input

                # convolution
                W = get_weight_variable('filter_gate_W', (fw, r, 2*r))
                b = get_bias_variable('filter_gate_b', (2*r))
                Y = tf.nn.convolution(
                    X, W, padding='SAME', dilation_rate=[dilation])
                Y += b

                if self._use_cond:
                    W = get_weight_variable(
                        'cond_filter_gate_W', (1, self.cond_embed_dim, 2*r))
                    b = get_bias_variable('cond_filter_gate_b', (2*r))
                    C = tf.nn.convolution(
                        cond_input, W, padding='SAME')  # 1x1 convolution
                    C += b
                    Y += C

                if self.use_dropout:
                    Y = tf.nn.dropout(Y, keep_prob=self.dropout_keep_prob)

                # filter and gate
                Y = tf.tanh(Y[:, :, :r])*tf.sigmoid(Y[:, :, r:])
                #Y = Y[:, :, :r]*tf.sigmoid(Y[:, :, r:]) # GLU

                # skip 1x1 convolution
                W = get_weight_variable('skip_W', (1, r, s))
                b = get_bias_variable('skip_b', (s))
                skip_out = tf.nn.convolution(X, W, padding='SAME')
                skip_out += b

                # output 1x1 convolution
                W = get_weight_variable('output_W', (1, r, r))
                b = get_bias_variable('output_b', r)
                Y = tf.nn.convolution(Y, W, padding='SAME')
                Y += b

                Y += X

        return Y, skip_out

    def _postproc_module(self, skip_outputs):
        with tf.variable_scope('postproc_module'):

            s = self.postnet_channels

            # sum of residual module outputs
            X = tf.zeros_like(skip_outputs[0])
            for R in skip_outputs:
                #R = tf.nn.relu(R) # activation maybe
                X += R

            # 1x1 convolution
            W1 = get_weight_variable('W1', shape=(1, s, s))
            b1 = get_bias_variable('b1', shape=s)
            Y = tf.nn.convolution(X, W1, padding='SAME')
            Y += b1

            Y = tf.nn.tanh(Y)

            # final output
            if type(self.output_channels) is list:
                output_list = []
                for i, c in enumerate(self.output_channels):
                    W = get_weight_variable('W_out{}'.format(i), (1, s, c))
                    b = get_bias_variable('b_out{}'.format(i), c)
                    out = tf.nn.convolution(Y, W, padding='SAME')
                    out += b
                    output_list.append(out)
                Y = output_list
            else:
                W = get_weight_variable('W_out', (1, s, self.output_channels))
                b = get_bias_variable('b_out', self.output_channels)
                out = tf.nn.convolution(Y, W, padding='SAME')
                out += b
                Y = out

        return Y

    def forward_pass(self, X_input, cond_input=None):

        skip_outputs = []
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):

            if self._use_cond:
                C = self._embed_cond(cond_input)
            else:
                C = None

            R = self._input_layer(X_input)
            X = R
            for i, dilation in enumerate(self.dilations):
                X, skip = self._conv_module(X, R, i, dilation, cond_input=C)
                skip_outputs.append(skip)

            Y = self._postproc_module(skip_outputs)

        return Y
