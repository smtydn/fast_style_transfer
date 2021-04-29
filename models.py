import tensorflow as tf
import keras
from keras import layers
from keras.applications import vgg16
import tensorflow_addons as tfa
import utils


def style_transfer_net(width=256, height=256, bs=32, training=False):
    inputs = layers.Input(shape=(width, height, 3), dtype='float32')

    ##################
    # ENCODING BLOCK #
    ##################
    # BLOCK 1
    # Convolution: 32 filters, kernel size (9, 9), 1 stride
    # Instance Normalization
    # ReLU activation
    eb1 = _conv_block(inputs, 32, (9, 9), 1)

    # BLOCK 2
    # Convolution: 64 filters, kernel size (3, 3), 2 strides
    # Instance Normalization
    # ReLU activation
    eb2 = _conv_block(eb1, 64, (3, 3), 2)

    # BLOCK 3
    # Reflection padding: 1
    # Convolution: 128 filters, kernel size (3, 3), 2 strides
    # Instance Normalization
    # ReLU activation
    eb2 = _conv_block(eb2, 128, (3, 3), 2)

    ###################
    # RESIDUAL BLOCKS #
    ###################
    # For any single residual block, there exists:
    # Convolution: 128 filters, kernel size (3, 3), 1 stride
    # Instance Normalization
    # ReLU activation
    # Convolution: 128 filters, kernel size (3, 3), 1 stride
    # Instance Normalization
    # Merge block which merges the input and the result of previous layers
    rb1 = _residual_block(eb2)
    rb2 = _residual_block(rb1)
    rb3 = _residual_block(rb2)
    rb4 = _residual_block(rb3)
    rb5 = _residual_block(rb4)

    ###################
    # DECODING BLOCKS #
    ###################
    # BLOCK 1
    # Transpose Convolution: 64 filters, kernel size (3, 3), 2 strides
    # Instance Normalization
    # ReLU activation
    db1 = _conv_transpose_block(rb5, 64, (3, 3), 2)
    
    # BLOCK 2
    # Transpose Convolution: 32 filters, kernel size (3, 3), 2 stride
    # Instance Normalization
    # ReLU activation
    db2 = _conv_transpose_block(db1, 32, (3, 3), 2)

    # BLOCK 3
    # Convolution: 3 filters, kernel size (9, 9), 1 strides
    # Instance Normalization
    db3 = _conv_block(db2, 3, (9, 9), 1, activation='tanh')
    
    # Scale output
    outputs = layers.Lambda(lambda x: x * 150 + 255./2, name='output')(db3)

    # Attach the VGG16 network while training
    if training:
        # VGG
        content_activation = layers.Input(shape=(width // 2, height // 2, 128), dtype='float32', name='content_activation')
        style_activation1 = layers.Input(shape=(width, height, 64), dtype='float32', name='style_activation1')
        style_activation2 = layers.Input(shape=(width // 2, height // 2, 128), dtype='float32', name='style_activation2')
        style_activation3 = layers.Input(shape=(width // 4, height // 4, 256), dtype='float32', name='style_activation3')
        style_activation4 = layers.Input(shape=(width // 8, height // 8, 512), dtype='float32', name='style_activation4')

        total_variation_loss = layers.Lambda(utils.get_total_var_loss, output_shape=(1,), arguments={'height': height, 'width': width}, name='tv')([outputs])

        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(outputs)
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
            [content_loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, outputs]
        )
        model_layers = {layer.name: layer for layer in model.layers}
        original_vgg = vgg16.VGG16(weights='imagenet', include_top=False)
        original_vgg_layers = {layer.name: layer for layer in original_vgg.layers}

        # load image_net weight
        for layer in original_vgg.layers:
            if layer.name in model_layers:
                model_layers[layer.name].set_weights(original_vgg_layers[layer.name].get_weights())
                model_layers[layer.name].trainable = False
    else:
        model = keras.Model(inputs, outputs)

    return model


def _conv_block(x, filters, kernel_size, strides, activation='relu'):
    conv = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    instance_norm = tfa.layers.InstanceNormalization()(conv)

    if activation == 'relu':
        return layers.ReLU()(instance_norm)
    elif activation == 'tanh':
        return layers.Activation(tf.nn.tanh)(instance_norm)
    else:
        return instance_norm


def _conv_transpose_block(x, filters, kernel_size, strides):
    conv_tp = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same')(x)
    instance_norm = tfa.layers.InstanceNormalization()(conv_tp)
    return layers.ReLU()(instance_norm)



def _residual_block(x):
    res = x
    x = _conv_block(x, 128, (3, 3), strides=1)
    x = _conv_block(x, 128, (3, 3), strides=1, activation=None)
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
