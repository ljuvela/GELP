import tensorflow as tf

_FLOATX = tf.float32

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


class ConvModel:

    def __init__(self, name, in_channels, filter_width=9, n_hidden=[64]):
        self.filter_width = filter_width
        self.n_hidden = n_hidden
        self.in_channels = in_channels
        self._name = name

    def _conv_layer(self, X, filter_width, layer_id, in_channels, out_channels):
        with tf.variable_scope('conv_layer{}'.format(layer_id)):
            W = get_weight_variable(
                'W', (filter_width, in_channels, out_channels))
            b = get_bias_variable('b', (out_channels))
            Y = tf.nn.convolution(X, W, padding='SAME')  # 1x1 convolution
            Y += b
            return tf.tanh(Y)

    def forward_pass(self, X_input):
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            h = X_input
            in_channels = self.in_channels
            for i, out_channels in enumerate(self.n_hidden):
                h = self._conv_layer(h, self.filter_width,
                                     i, in_channels, out_channels)
                in_channels = out_channels

        return h

    def get_variable_list(self):
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
        if len(theta) == 0:
            raise RuntimeError("Model '{}' variabiles are not allocated until the first call of forward_pass".format(self._name))         
        return theta 

class UpsampleBilinearInterp():

    def __init__(self, upsample_factor=80, channels=128):
        self.upsample_factor = upsample_factor
        self.channels = channels

    def forward_pass(self, X):     
        C = self.channels # number of channels
        K = self.upsample_factor # upsampling factor
        X = tf.expand_dims(X, axis=1)
        size = tf.stack([1, K*tf.shape(X)[2]], axis=0)
        Y = tf.image.resize (X, size=size, method=tf.image.ResizeMethod.BILINEAR)
        Y = tf.squeeze(Y, axis=1) # remove height dim
        return Y  
