import tensorflow as tf
from keras import Model, Sequential
from keras.applications import vgg16
from keras.layers import Activation, Lambda
from keras.activations import tanh

import settings
from src.blocks import ConvBlock, ResidualBlock, TransposeConvBlock


class TransformNet(Model):
    """ Image Transform Network """
    def __init__(self):
        super(TransformNet, self).__init__()
        self.model = Sequential([
            ConvBlock(32, 9, 1),
            ConvBlock(64, 3, 2),
            ConvBlock(128, 3, 2),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            TransposeConvBlock(64, 3, 2),
            TransposeConvBlock(32, 3, 2),
            ConvBlock(3, 9, 1),
            Activation(tanh),
            Lambda(lambda x: x * 150 + 255./2)
        ])

    def call(self, x):
        return self.model(x)


class LossNet(Model):
    """ Pretrained VGG16 network. Returns needed intermediate layers for style and content losses. """
    def __init__(self):
        super(LossNet, self).__init__()
        self.style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
        self.content_layers = ['block3_conv3']
        self.orig_vgg = vgg16.VGG16(include_top=False, weights=settings.VGG16_WEIGHTS_PATH)
        self.orig_vgg.trainable = False
        self.vgg = tf.keras.Model(
            [self.orig_vgg.input], 
            [self.orig_vgg.get_layer(layer_name).output for layer_name in self.style_layers + self.content_layers]
        )
        self.vgg.trainable = False

    def call(self, x):
        vgg_outputs = self.vgg(x)
        style_features, content_features = (vgg_outputs[:len(self.style_layers)], vgg_outputs[len(self.style_layers):])
        return {'style': style_features, 'content': content_features}
