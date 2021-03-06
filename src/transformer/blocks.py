import tensorflow as tf
import tensorflow_addons as tfa


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, relu=True):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')
        self.norm = tfa.layers.InstanceNormalization()
        self.relu = relu

    def call(self, x):
        x = self.norm(self.conv(x))
        if self.relu:
            x = tf.keras.layers.ReLU()(x)
        return x


class TransposeConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(TransposeConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')
        self.norm = tfa.layers.InstanceNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        return self.relu(self.norm(self.conv(x)))


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(128, 3, 1)
        self.conv2 = ConvBlock(128, 3, 1, relu=False)

    def call(self, x):
        res = x
        return res + self.conv2(self.conv1(x))