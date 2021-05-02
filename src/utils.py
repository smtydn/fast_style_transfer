import numpy as np
import tensorflow as tf
from keras.applications import vgg16


def load_image(image_path, target_size=None):
    """ Load image to be able to feed it to the model """
    # Load image from the path, resize if it needed
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    # Convert image to a Numpy array
    img = tf.keras.preprocessing.image.img_to_array(img, dtype='float32')
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def gram_matrix(x):
    """ Calculate gram matrix for given input tensor """
    # Calculate the summation part of the formula
    result = tf.einsum('bijc,bijd->bcd', x, x)
    # Divide summation part to (channels * height * width)
    # Given tensor's shape: (batch_size, height, width, channels)
    return result / (x.shape[1] * x.shape[2] * x.shape[3])
