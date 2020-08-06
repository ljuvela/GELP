from __future__ import division, print_function

__author__ = "Lauri Juvela, lauri.juvela@aalto.fi"

import os
import sys
import math
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    try:
        tf.enable_eager_execution()
    except ValueError as e:
        if e.args[0] != 'tf.enable_eager_execution must be called at program startup.':
            raise e

_FLOATX = tf.float32 # todo: move to lib/precision.py

def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    if shape is None:
        return tf.get_variable(name)
    else:  
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX)): 
    if shape is None:
        return tf.get_variable(name) 
    else:     
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)
   

class Wavenet():

    def __init__(self,
                 name,
                 residual_channels=64,
                 filter_width=3,
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 input_channels=123,
                 output_channels=48,
                 cond_dim = None,
                 cond_channels = 64,
                 postnet_channels=256):

        if filter_width % 2 != 1:
            raise ValueError("filter_width must be odd")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.postnet_channels = postnet_channels

        self.leaky_relu_alpha = 0.1
       
        if cond_dim is not None:
            self._use_cond = True
            self.cond_dim = cond_dim
            self.cond_channels = cond_channels
            
        else:
            self._use_cond = False

        self._name = name
        self._create_variables()

    def get_receptive_field(self):
        # returns past receptive field length (not including current time)
        fw = self.filter_width
        d = self.dilations
        # input layer + dilated layers (post-proc layers are 1x1) 
        R = (1 + sum(d)) * (fw - 1) // 2
        return R
        
    def _create_variables(self):

        fw = self.filter_width
        r = self.residual_channels
        s = self.postnet_channels

        with tf.variable_scope(self._name):

            with tf.variable_scope('input_layer'):
                get_weight_variable('W', (fw, self.input_channels, r))
                get_bias_variable('b', (r)) 

            if self._use_cond:
                with tf.variable_scope('embed_cond'):
                    get_weight_variable('W', (1, self.cond_dim, self.cond_channels))
                    get_bias_variable('b', (self.cond_channels))         

            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('conv_modules'):
                    with tf.variable_scope('module{}'.format(i)):
                        get_weight_variable('filter_gate_W', (fw, r, 2*r)) 
                        get_bias_variable('filter_gate_b', (2*r)) 

                        if self._use_cond:
                            get_weight_variable('cond_filter_gate_W', (1, self.cond_channels, 2*r)) 
                            get_bias_variable('cond_filter_gate_b', (2*r)) 
                                            
            with tf.variable_scope('postproc_module'):
                
                self._postnet_hidden = None # initialize before forward pass

                get_weight_variable('W1', (1, r, s)) 
                get_bias_variable('b1', s)

                if type(self.output_channels) is list:
                    get_weight_variable('W2', (1, s, sum(self.output_channels))) 
                    get_bias_variable('b2', sum(self.output_channels))
                else:
                    get_weight_variable('W2', (1, s, self.output_channels)) 
                    get_bias_variable('b2', self.output_channels)

    def get_variable_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)          

    def get_postnet_hidden(self):
        return self._postnet_hidden

    def _input_layer(self, main_input):
        with tf.variable_scope('input_layer'):

            r = self.residual_channels
            fw = self.filter_width
            W = get_weight_variable('W', (fw, self.input_channels, r))
            b = get_bias_variable('b',(r))

            X = main_input

            Y = tf.nn.convolution(X, W, padding='VALID')
            Y += b
            Y = tf.tanh(Y)

        return Y

    def _embed_cond(self, cond_input):
        with tf.variable_scope('embed_cond'):
            W = get_weight_variable('W', (1, self.cond_dim, self.cond_channels))
            b = get_bias_variable('b',  (self.cond_channels))

            Y = tf.nn.convolution(cond_input, W, padding='VALID') # 1x1 convolution
            Y += b
            Y = tf.nn.tanh(Y)

            return Y

    def _conv_module(self, main_input, residual_input, module_idx, dilation, cond_input=None):
        with tf.variable_scope('conv_modules'):
            with tf.variable_scope('module{}'.format(module_idx)):
                r = self.residual_channels
                fw = self.filter_width
                trim = dilation * (fw - 1) // 2
                
                X = main_input

                # convolution
                W = get_weight_variable('filter_gate_W', (fw, r, 2*r)) 
                b = get_bias_variable('filter_gate_b', (2*r)) 
                Y = tf.nn.convolution(X, W, padding='VALID', dilation_rate=[dilation])
                # add bias
                Y += b

                if self._use_cond:
                    V_cond = get_weight_variable('cond_filter_gate_W', (1, self.cond_channels, 2*r)) 
                    b_cond = get_bias_variable('cond_filter_gate_b', (2*r)) 
                    cond_trimmed = cond_input[:,trim:-trim]
                    C = tf.nn.convolution(cond_trimmed, V_cond, padding='VALID') # 1x1 convolution
                    C += b_cond
                    Y += C

                # filter and gate
                Y = tf.tanh(Y[:, :, :r])*tf.sigmoid(Y[:, :, r:])

        if self._use_cond:
            return Y, cond_trimmed
        else:
            return Y

    def _postproc_module(self, X):
        with tf.variable_scope('postproc_module'):

            fw = self.filter_width
            r = self.residual_channels
            s = self.postnet_channels
            if type(self.output_channels) is list:
                o = sum(self.output_channels)
            else:     
                o = self.output_channels

            W1 = get_weight_variable('W1', (1, r, s)) # 1x1 conv
            b1 = get_bias_variable('b1', s)
            Y = tf.nn.convolution(X, W1, padding='VALID')    
            Y += b1
            Y = tf.nn.tanh(Y)
            
            # for feature matching
            self._postnet_hidden = Y
      
            W2 = get_weight_variable('W2', (1, s, o)) # 1x1 conv
            b2 = get_bias_variable('b2', o)
            Y = tf.nn.convolution(Y, W2, padding='VALID')    
            Y += b2

            if type(self.output_channels) is list:
                output_list = []
                start = 0 
                for channels in self.output_channels:
                    output_list.append(Y[:,:,start:start+channels])
                    start += channels
                Y = output_list
            
        return Y

    def forward_pass(self, X_input, cond_input=None):
        
        with tf.variable_scope(self._name, reuse=True):

            if self._use_cond:
                C = self._embed_cond(cond_input)
                fw = self.filter_width
                trim = (fw - 1) // 2
                C = C[:,trim:-trim]
            
            X = self._input_layer(X_input)
            R = X
            for i, dilation in enumerate(self.dilations):
                if self._use_cond:
                    X, C = self._conv_module(X, R, i, dilation, cond_input=C)
                else:
                    X = self._conv_module(X, R, i, dilation, cond_input=None)    
            
            Y = self._postproc_module(X)    

        return Y
