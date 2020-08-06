from __future__ import division, print_function

__author__ = "Lauri Juvela, lauri.juvela@aalto.fi"

import os
import sys
import math
import numpy as np
import tensorflow as tf

_FLOATX = tf.float32 # todo: move to lib/precision.py

def get_weight_variable(name, shape=None, initial_value=None):
    if shape is None:
        return tf.get_variable(name)

    if initial_value is None:
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        W = tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)
    else:
        W = tf.Variable(initial_value)

    return W

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX)):
    return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)

def convolution(X, W, dilation=1, causal=True):
    """
    Applies 1D convolution

    Args:
        X: input tensor of shape (batch, timesteps, in_channels)
        W: weight tensor of shape (filter_width, in_channels, out_channels)
        dilation: int value for dilation
        causal: bool flag for causal convolution

    Returns:
        Y: output tensor of shape (batch, timesteps, out_channels)

    """

    if causal:
        fw = tf.shape(W)[0]
        pad = (fw - 1) * dilation
        Y = tf.pad(X, paddings=[[0,0], [pad,0], [0,0]])
        Y = tf.nn.convolution(Y, W, padding="VALID", dilation_rate=[dilation])
    else:
        Y = tf.nn.convolution(X, W, padding="SAME", dilation_rate=[dilation])
    
    return Y

class WaveNet():
    """
    TensorFlow WaveNet object

    Initialization Args:
        name: string used for variable namespacing
            user is responsible for unique names if multiple models are used

        residual_channels: number of channels used in the convolution layers

        postnet_channels: 

        filter_width: 

        dilations: list of integers containing the dilation pattern
            list length determines the number of dilated blocks used

        input_channels:

        causal: if True, use causal convolutions everywhere in the network

        conv_block_gate: if True, use gated convolutions in the dilated blocks

        conv_block_affine_out: if True, apply a 1x1 convolution in dilated blocks before the residual connection     

    Functions:

    Members:

    """

    def __init__(self,
                 name,
                 residual_channels=64,
                 postnet_channels=64,
                 filter_width=3,
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 input_channels=1,
                 output_channels=1,
                 cond_channels=None,
                 cond_embed_dim = 64,
                 causal=True,
                 conv_block_gate=True,
                 conv_block_affine_out=True,
                 add_noise_at_each_layer=False
                ):

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.postnet_channels = postnet_channels

        self.causal = causal
        self.conv_block_gate = conv_block_gate
        self.conv_block_affine_out = conv_block_affine_out
        self.add_noise_at_each_layer = add_noise_at_each_layer

        if cond_channels is not None:
            self._use_cond = True
            self.cond_embed_dim = cond_embed_dim
            self.cond_channels = cond_channels
        else:
            self._use_cond = False

        self._name = name

    def get_receptive_field(self):
        receptive_field = (self.filter_width - 1) * sum(self.dilations) + 1 # due to dilation layers
        receptive_field += self.filter_width - 1                       # due to input layer (if not 1x1)
        if not self.causal:
            receptive_field = 2 * receptive_field - 1 
        return receptive_field

    def get_variable_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)

    def _input_layer(self, main_input):
        with tf.variable_scope('input_layer'):

            r = self.residual_channels
            fw = self.filter_width
            W = get_weight_variable('W', (fw, self.input_channels, r))
            b = get_bias_variable('b', (r))

            X = main_input

            Y = convolution(X, W, causal=self.causal)
            Y += b
            Y = tf.tanh(Y)

        return Y

    def _embed_cond(self, cond_input):
        with tf.variable_scope('embed_cond'):
            W = get_weight_variable('W', (1, self.cond_channels, self.cond_embed_dim))
            b = get_bias_variable('b', (self.cond_embed_dim))

            Y = convolution(cond_input, W, causal=self.causal) # 1x1 convolution
            Y += b

            return tf.tanh(Y)

    def _conv_module(self, main_input, module_idx, dilation, cond_input=None):
        with tf.variable_scope('conv_modules'):
            with tf.variable_scope('module{}'.format(module_idx)):

                fw = self.filter_width
                r = self.residual_channels
                X = main_input

                if self.conv_block_gate:    
                    # convolution
                    W = get_weight_variable('filter_gate_W', (fw, r, 2*r))
                    b = get_bias_variable('filter_gate_b', (2*r))
                    Y = convolution(X, W,
                                    dilation=dilation,
                                    causal=self.causal)
                    Y += b

                    # conditioning
                    if self._use_cond:
                        V = get_weight_variable('cond_filter_gate_W',
                                                (1, self.cond_embed_dim, 2*r))
                        b = get_bias_variable('cond_filter_gate_b', (2*r))
                        C = convolution(cond_input, V) # 1x1 convolution
                        Y += C + b

                    if self.add_noise_at_each_layer:
                        W = get_weight_variable('noise_scaling_W',
                                            (1, 1, r))
                        Z = tf.random_normal(shape=tf.shape(Y[..., :r]))
                        Y += tf.concat([W * Z, tf.zeros_like(Y[..., r:])], axis=-1)

                    # filter and gate
                    Y = tf.tanh(Y[..., :r]) * tf.sigmoid(Y[..., r:])

                else:
                    # convolution
                    W = get_weight_variable('filter_gate_W', (fw, r, r))
                    b = get_bias_variable('filter_gate_b', (r))
                    Y = convolution(X, W,
                                    dilation=dilation,
                                    causal=self.causal)
                    Y += b

                    # conditioning
                    if self._use_cond:
                        V = get_weight_variable('cond_filter_gate_W',
                                                (1, self.cond_embed_dim, r))
                        b = get_bias_variable('cond_filter_gate_b', (r))
                        C = convolution(cond_input, V) # 1x1 convolution
                        Y += C + b

                    if self.add_noise_at_each_layer:
                        W = get_weight_variable('noise_scaling_W',
                                            (1, 1, r))
                        Z = tf.random_normal(shape=tf.shape(Y))
                        Y += W * Z

                    # activation
                    Y = tf.tanh(Y)

                skip_out = Y

                if self.conv_block_affine_out:
                    W = get_weight_variable('output_W', (1, r, r))
                    b = get_bias_variable('output_b', (r))
                    Y = convolution(Y, W) + b

                # residual connection
                Y += X

        return Y, skip_out

    def _postproc_module(self, residual_module_outputs):
        with tf.variable_scope('postproc_module'):

            s = self.postnet_channels
            r = self.residual_channels
            d = len(self.dilations)

            # concat and convolve
            W1 = get_weight_variable('W1', (1, d*r, s))
            b1 = get_bias_variable('b1', s)
            X = tf.concat(residual_module_outputs, axis=-1) # concat along channel dim
            Y = convolution(X, W1)
            Y += b1
            Y = tf.nn.tanh(Y)

            # output layer
            W2 = get_weight_variable('W2', (1, s, self.output_channels))
            b2 = get_bias_variable('b2', self.output_channels)
            Y = convolution(Y, W2)
            Y += b2

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
                X, skip = self._conv_module(X, i, dilation, cond_input=C)
                skip_outputs.append(skip)
            Y = self._postproc_module(skip_outputs)
        return Y

