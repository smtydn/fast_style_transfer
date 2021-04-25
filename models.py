import tensorflow as tf
import keras
from keras import layers
from keras.applications import vgg16
import tensorflow_addons as tfa
import utils


def image_transform_network(width=256, height=256):
    inputs = layers.Input(shape=(width, height, 3), dtype='float32')

    # encoding blocks
    # block 1: reflection pad 4, conv 32x9x9 with strides=1, instance normalization and relu
    pad_eb1 = tf.pad(inputs, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='REFLECT')
    conv_eb1 = layers.Conv2D(32, (9, 9), strides=1, padding='valid')(pad_eb1)
    in_eb1 = tfa.layers.InstanceNormalization()(conv_eb1)
    relu_eb1 = layers.ReLU()(in_eb1)
    # block 2: reflection pad 1, conv 64x3x3 with strides=2, instance normalization and relu
    pad_eb2 = tf.pad(relu_eb1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    conv_eb2 = layers.Conv2D(64, (3, 3), strides=2, padding='valid')(pad_eb2)
    in_eb2 = tfa.layers.InstanceNormalization()(conv_eb2)
    relu_eb2 = layers.ReLU()(in_eb2)
    # block 2: reflection pad 1, conv 128x3x3 with strides=2, instance normalization and relu
    pad_eb3 = tf.pad(relu_eb2, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    conv_eb3 = layers.Conv2D(128, (3, 3), strides=2, padding='valid')(pad_eb3)
    in_eb3 = tfa.layers.InstanceNormalization()(conv_eb3)
    relu_eb3 = layers.ReLU()(in_eb3)

    # residual blocks
    res1 = _resblock(relu_eb3)
    res2 = _resblock(res1)
    res3 = _resblock(res2)
    res4 = _resblock(res3)
    res5 = _resblock(res4)

    # decoding blocks
    # block 1: upsample by factor 2, apply refpad, conv2d 64x3x3 strides=1 and relu
    upsample_db1 = layers.UpSampling2D(size=(2, 2))(res5)
    pad_db1 = tf.pad(upsample_db1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    conv_db1 = layers.Conv2D(64, (3, 3), strides=1, padding='valid')(pad_db1)
    relu_db1 = layers.ReLU()(conv_db1)
    
    # block2: upsample by factor 2, apply refpad, conv2d 32x3x3 strides=1 and relu
    upsample_db2 = layers.UpSampling2D(size=(2, 2))(relu_db1)
    pad_db2 = tf.pad(upsample_db2, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    conv_db2 = layers.Conv2D(32, (3, 3), strides=1, padding='valid')(pad_db2)
    relu_db2 = layers.ReLU()(conv_db2)

    # block3: refpad 4, conv2d 3x9x9 strides=1, tanh
    pad_db3 = tf.pad(upsample_db2, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='REFLECT')
    conv_db3 = layers.Conv2D(3, (9, 9), strides=1, padding='valid')(pad_db3)
    relu_db3 = layers.Activation(tf.keras.activations.tanh)(conv_db3)
    
    # scale output from [-1, 1] to [0, 255]
    outputs = layers.Lambda(lambda x: (x + 1) * 127.5)(relu_db3)

    return keras.Model(inputs, outputs)


# TODO: get_total_var_loss, get_style_loss, get_content_loss
def loss_network(width=256, height=256, bs=32):
    inputs = layers.Input(shape=(height, width, 3), dtype='float32', name='input_1')

    content_activation = layers.Input(shape=(width // 2, height // 2, 128), dtype='float32', name='content_activation')
    style_activation1 = layers.Input(shape=(width, height, 64), dtype='float32', name='style_activation1')
    style_activation2 = layers.Input(shape=(width // 2, height // 2, 128), dtype='float32', name='style_activation2')
    style_activation3 = layers.Input(shape=(width // 4, height // 4, 256), dtype='float32', name='style_activation3')
    style_activation4 = layers.Input(shape=(width // 8, height // 8, 512), dtype='float32', name='style_activation4')

    total_variation_loss = layers.Lambda(utils.get_total_var_loss, output_shape=(1,), arguments={'height': height, 'width': width})([inputs])

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    # style_loss1 = layers.Lambda(utils.get_style_loss, output_shape=(1,),
    #                             name='style1', arguments={'batch_size': bs})([x, style_activation1])
    style_loss1 = StyleReconstructionLossLayer(name='style1')(new_activation=x, style_activation=style_activation1, batch_size=bs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    # content_loss = layers.Lambda(utils.get_content_loss, output_shape=(1,), name='content')([x, content_activation])
    # style_loss2 = layers.Lambda(utils.get_style_loss, output_shape=(1,),
    #                             name='style2', arguments={'batch_size': bs})([x, style_activation2])
    content_loss = FeatureReconstructionLossLayer(name='content')(x, content_activation)
    style_loss2 = StyleReconstructionLossLayer(name='style2')(new_activation=x, style_activation=style_activation2, batch_size=bs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    # style_loss3 = layers.Lambda(utils.get_style_loss, output_shape=(1,),
    #                             name='style3', arguments={'batch_size': bs})([x, style_activation3])
    style_loss3 = StyleReconstructionLossLayer(name='style3')(new_activation=x, style_activation=style_activation3, batch_size=bs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    style_loss4 = StyleReconstructionLossLayer(name='style4')(new_activation=x, style_activation=style_activation4, batch_size=bs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = keras.Model(
        [inputs, content_activation, style_activation1, style_activation2, style_activation3, style_activation4], 
        [content_loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss]
    )
    model_layers = {layer.name: layer for layer in model.layers}
    original_vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    original_vgg_layers = {layer.name: layer for layer in original_vgg.layers}

    # load image_net weight
    for layer in original_vgg.layers:
        if layer.name in model_layers:
            model_layers[layer.name].set_weights(original_vgg_layers[layer.name].get_weights())
            model_layers[layer.name].trainable = False

    return model


def _resblock(x):
    res = x
    x = layers.Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    return layers.Add()([x, res])


class StyleReconstructionLossLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(StyleReconstructionLossLayer, self).__init__(*args, **kwargs)
        self.fn = utils.get_style_loss

    def call(self, batch_size, new_activation, style_activation):
        return self.fn(batch_size, new_activation, style_activation)


class FeatureReconstructionLossLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(FeatureReconstructionLossLayer, self).__init__(*args, **kwargs)
        self.fn = utils.get_content_loss

    def call(self, new_activation, content_activation):
        return self.fn(new_activation, content_activation)
