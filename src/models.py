import keras
import tensorflow as tf
import tensorflow_addons as tfa
from keras.applications import vgg16

import blocks


class TransformNet(keras.Model):
    def __init__(self):
        super(TransformNet, self).__init__()
        self.conv1 = blocks.ConvBlock(32, 9, 1)
        self.conv2 = blocks.ConvBlock(64, 3, 2)
        self.conv3 = blocks.ConvBlock(128, 3, 2)
        self.res1 = blocks.ResidualBlock()
        self.res2 = blocks.ResidualBlock()
        self.res3 = blocks.ResidualBlock()
        self.res4 = blocks.ResidualBlock()
        self.res5 = blocks.ResidualBlock()
        self.conv4 = blocks.TransposeConvBlock(64, 3, 2)
        self.conv5 = blocks.TransposeConvBlock(32, 3, 2)
        self.conv6 = blocks.ConvBlock(3, 9, 1)
        self.activation = keras.layers.Activation(keras.activations.tanh)
        self.normalize = keras.layers.Lambda(lambda x: x * 150 + 255./2)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = self.normalize(x)
        return x


class LossNet(keras.Model):
    def __init__(self):
        super(LossNet, self).__init__()
        self.loss_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
        self.orig_vgg = vgg16.VGG16(include_top=False, weights='imagenet')
        self.orig_vgg.trainable = False
        self.vgg = tf.keras.Model(
            [self.orig_vgg.input], 
            [self.orig_vgg.get_layer(layer_name).output for layer_name in self.loss_layers]
        )
        self.vgg.trainable = False

    def call(self, x):
        outputs = self.vgg(x)
        return dict(zip(self.loss_layers, outputs))
