import tensorflow as tf
from tensorflow.keras.applications import vgg16

from src import settings
from src.transformer import blocks


class TransformNet(tf.keras.Model):
    """ Image Transform Network """
    def __init__(self):
        super(TransformNet, self).__init__()
        self.model = tf.keras.Sequential([
            blocks.ConvBlock(32, 9, 1),
            blocks.ConvBlock(64, 3, 2),
            blocks.ConvBlock(128, 3, 2),
            blocks.ResidualBlock(),
            blocks.ResidualBlock(),
            blocks.ResidualBlock(),
            blocks.ResidualBlock(),
            blocks.ResidualBlock(),
            blocks.TransposeConvBlock(64, 3, 2),
            blocks.TransposeConvBlock(32, 3, 2),
            blocks.ConvBlock(3, 9, 1),
            tf.keras.layers.Activation(tf.keras.activations.tanh),
            tf.keras.layers.Lambda(lambda x: (x + 1) * 127.5)
        ])

    def call(self, x):
        return self.model(x)


class LossNet(tf.keras.Model):
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
        x = vgg16.preprocess_input(x)
        vgg_outputs = self.vgg(x)
        style_features, content_features = (vgg_outputs[:len(self.style_layers)], vgg_outputs[len(self.style_layers):])
        return {'style': style_features, 'content': content_features}
