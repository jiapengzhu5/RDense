import tensorflow as tf
import numpy as np

def conv_layer(input, filter, kernel, stride=1, layer_name='conv'):
    with tf.name_scope(layer_name):
        net = tf.layers.conv2d(inputs=input,
                               use_bias=False,
                               filters=filter,
                               kernel_size=kernel,
                               strides=stride,
                               padding='SAME')
        return net

def Global_average_pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter

def Batch_norm(x, training, scope):
    return tf.cond(
        training,
        lambda : tf.layers.batch_normalization(
            inputs=x,
            trainable=True,
            reuse=None,
            name=scope),
        lambda : tf.layers.batch_normalization(
            inputs=x,
            trainable=False,
            reuse=True,
            name=scope)
    )

def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers):
    return tf.concat(layers, axis=3)

class DenseNet():
    def __init__(self, x, nb_blocks, filters, training, dropout_rate):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.dropout_rate = dropout_rate
        self.logits = self.densenet(x)

    def bottle_neck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_norm(x, training=self.training, scope='%s_batch1' % scope)
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name= '%s_conv1' % scope)
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            x = Batch_norm(x, training=self.training, scope='%s_batch2' % scope)
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name='%s_batch2' % scope)
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_norm(x, training=self.training, scope='%s_batch1' % scope)
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name='%s_conv1' % scope)
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 1], stride=1)
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layer_concat = list()
            layer_concat.append(input_x)

            x = self.bottle_neck_layer(input_x, scope='%s_bottleN_%d' % (layer_name, 0))

            layer_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layer_concat)
                x = self.bottle_neck_layer(x, scope='%s_bottleN_%d' % (layer_name, i + 1))
                layer_concat.append(x)

            x = Concatenation(layer_concat)
            return x

    def densenet(self, input_x):
        x = conv_layer(input_x, filter= 2 * self.filters, kernel=[5, 5], stride=2, layer_name='conv0')
        x = self.dense_block(input_x=x, nb_layers=3, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=3, layer_name='dense_3')
        x = Batch_norm(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_average_pooling(x)
        return x