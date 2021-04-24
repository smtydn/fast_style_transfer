import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


def transform_net(input_tensor):
    # encoding block
    econv1 = encoding_layer(filters=32, kernel_size=(9, 9), strides=1)(input_tensor)
    econv2 = encoding_layer(filters=64, kernel_size=(3, 3), strides=2)(econv1)
    econv3 = encoding_layer(filters=128, kernel_size=(3, 3), strides=2)(econv2)

    # residual blocks
    res1 = residual_block(econv3)
    res2 = residual_block(res1)
    res3 = residual_block(res2)
    res4 = residual_block(res3)
    res5 = residual_block(res4)

    # decoding block
    deconv1 = decoding_layer(filters=64, kernel_size=(3, 3), strides=2)(res5)
    deconv2 = decoding_layer(filters=32, kernel_size=(3, 3), strides=2)(deconv1)
    deconv3 = decoding_layer(filters=3, kernel_size=(9, 9), strides=1)(deconv2)

    return tf.keras.Model(inputs=input_tensor, outputs=deconv2, name='Image Transform Network')


def encoding_layer(filters, kernel_size, strides):
    return tf.keras.Sequential([
        layers.Conv2D(filters, kernel_size, strides, padding='same'),
        tfa.layers.InstanceNormalization(),
        layers.ReLU()
    ])


def residual_block(input_tensor):
    res = conv_layer(filters=128, kernel_size=(3, 3), strides=1)(input_tensor)
    return res + conv_layer(filters=128, kernel_size=(3, 3), strides=1, activation=None)(input_tensor)


def decoding_layer(filters, kernel_size, strides, activation='relu'):
    model = tf.keras.Sequential([
        layers.Conv2DTranspose(filters, kernel_size, strides, padding='same'),
        tfa.layers.InstanceNormalization()
    ])

    if activation == 'relu':
        model.add(layers.Activation(tf.keras.activations.relu))
    elif activation == 'tanh':
        model.add(layers.Activation(tf.keras.activations.tanh))

    return model
