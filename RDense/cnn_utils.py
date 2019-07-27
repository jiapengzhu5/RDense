import numpy as np
import tensorflow as tf
import random
from tensorflow.contrib import rnn
# import numpy as np

def BiRNN(x, n_hidden, lengths,reuse = None, name=None):
        # name_scope('word_encoder'+name)
    with tf.variable_scope(name):
        lstm_fw_cell = rnn.GRUCell(n_hidden,reuse=reuse)
        lstm_bw_cell = rnn.GRUCell(n_hidden,  reuse=reuse)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32, sequence_length=lengths)
    return tf.concat((output_states[0], output_states[1]), axis=1)

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [41, 5], 4)
    return batch

def create_cnn_layer(data, weights_matrix, bias_vector, strides_x_y):
    all_strides = [1, strides_x_y[0], strides_x_y[1], 1]
    result = tf.nn.conv2d(data, weights_matrix, strides=all_strides, padding='VALID')
    result = tf.nn.bias_add(result, bias_vector)
    result = tf.nn.relu(result)
    return result

def create_pooling_layer(data, kpool_x_y, strides_x_y):
    all_kpools = [1, kpool_x_y[0], kpool_x_y[1], 1]
    all_strides = [1, strides_x_y[0], strides_x_y[1], 1]
    result = tf.nn.max_pool(data, ksize=all_kpools, strides=all_strides, padding='VALID')
    return result

def create_layer(data, weights_matrix, bias_vector, strides_x_y, kpool_x_y):
    result = create_cnn_layer(data, weights_matrix, bias_vector, strides_x_y)
    result = create_pooling_layer(result, kpool_x_y, strides_x_y)
    return result

def flatten(conv_data, fc_size):
    flat_data = tf.reshape(conv_data, [-1, fc_size])
    return flat_data

def nn_layer(data, weights, bias, activate_non_linearity):
    result = tf.add(tf.matmul(data, weights), bias)
    if activate_non_linearity:
        result = tf.nn.relu(result)
    return result

def pearson_correlation(x, y):
    mean_x, var_x = tf.nn.moments(x, [0])
    mean_y, var_y = tf.nn.moments(y, [0])
    std_x = tf.sqrt(var_x)
    std_y = tf.sqrt(var_y)
    mul_vec = tf.multiply((x - mean_x), (y - mean_y))
    covariance_x_y, _ = tf.nn.moments(mul_vec, [0])
    pearson = covariance_x_y / (std_x * std_y)
    return pearson

