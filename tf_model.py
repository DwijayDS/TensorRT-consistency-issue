import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def resblock(inputs, filters):
    x = layers.Conv2D(filters=filters, kernel_size=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=filters * 4, kernel_size=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    if inputs.shape[-1] != filters * 4:
        inputs = layers.Conv2D(filters=filters * 4, kernel_size=1, padding="same", use_bias=False)(inputs)
        inputs = layers.BatchNormalization(momentum=0.9)(inputs)
    x = x + inputs
    x = layers.ReLU()(x)
    return x


def transition_branch(x, c_out):
    num_branch_in, num_branch_out = len(x), len(c_out)
    x = x + [x[-1] for _ in range(num_branch_out - num_branch_in)]  # padding the list x with x[-1]
    x_new = []
    for idx, (x_i, c_i) in enumerate(zip(x, c_out)):
        if idx < num_branch_in:
            if x_i.shape[-1] != c_i:
                x_i = layers.Conv2D(filters=c_i, kernel_size=3, padding="same", use_bias=False)(x_i)
                x_i = layers.BatchNormalization(momentum=0.9)(x_i)
                x_i = layers.ReLU()(x_i)
        else:
            filter_in = x_i.shape[-1]
            for j in range(idx + 1 - num_branch_in):
                filter_out = c_i if j == idx - num_branch_in else filter_in
                x_i = layers.Conv2D(filters=filter_out, kernel_size=3, strides=2, padding="same", use_bias=False)(x_i)
                x_i = layers.BatchNormalization(momentum=0.9)(x_i)
                x_i = layers.ReLU()(x_i)
        x_new.append(x_i)
    return x_new


def basic_block(inputs, filters):
    x = layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    if inputs.shape[-1] != filters:
        inputs = layers.Conv2D(filters=filters, kernel_size=1, padding="same", use_bias=False)(inputs)
        inputs = layers.BatchNormalization(momentum=0.9)(inputs)
    x = x + inputs
    x = layers.ReLU()(x)
    return x


def branch_convs(x, num_block, c_out):
    x_new = []
    for x_i, num_conv, c in zip(x, num_block, c_out):
        for _ in range(num_conv):
            x_i = basic_block(x_i, c)
        x_new.append(x_i)
    return x_new


def fuse_convs(x, c_out):
    x_new = []
    for idx_out, planes_out in enumerate(c_out):
        x_new_i = []
        for idx_in, x_i in enumerate(x):
            if idx_in > idx_out:
                x_i = layers.Conv2D(filters=planes_out, kernel_size=1, padding="same", use_bias=False)(x_i)
                x_i = layers.BatchNormalization(momentum=0.9)(x_i)
                x_i = layers.UpSampling2D(size=(2**(idx_in - idx_out), 2**(idx_in - idx_out)))(x_i)
            elif idx_in < idx_out:
                for _ in range(idx_out - idx_in - 1):
                    x_i = layers.Conv2D(x_i.shape[-1], kernel_size=3, strides=2, padding="same", use_bias=False)(x_i)
                    x_i = layers.BatchNormalization(momentum=0.9)(x_i)
                    x_i = layers.ReLU()(x_i)
                x_i = layers.Conv2D(planes_out, kernel_size=3, strides=2, padding="same", use_bias=False)(x_i)
                x_i = layers.BatchNormalization(momentum=0.9)(x_i)
            x_new_i.append(x_i)
        x_new.append(layers.ReLU()(tf.math.add_n(x_new_i)))
    return x_new


def hrstage(x, num_module, num_block, c_out):
    x = transition_branch(x, c_out)
    for _ in range(num_module):
        x = branch_convs(x, num_block, c_out)
        x = fuse_convs(x, c_out)
    return x


def hrnet(input_shape=(256, 256, 3), num_classes=17):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = resblock(x, 64)
    x = resblock(x, 64)
    x = resblock(x, 64)
    x = resblock(x, 64)
    x = hrstage([x], num_module=1, num_block=(4, 4), c_out=(32, 64))
    x = hrstage(x, num_module=4, num_block=(4, 4, 4), c_out=(32, 64, 128))
    x = hrstage(x, num_module=3, num_block=(4, 4, 4, 4), c_out=(32, 64, 128, 256))
    y = layers.Conv2D(filters=num_classes, kernel_size=1, activation="sigmoid")(x[0])
    z = layers.Conv2D(filters=num_classes, kernel_size=1, activation="sigmoid")(x[0])
    model = tf.keras.Model(inputs=inputs, outputs=[y,z])
    return model
