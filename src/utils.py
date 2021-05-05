import time
import os

import PIL
import numpy as np
import tensorflow as tf
from keras.applications import vgg16

import settings


def load_image_from_path(image_path, target_size=None):
    """ Load image to be able to feed it to the model """
    # Load image from the path, resize if it needed
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    # Convert image to a Numpy array
    img = tf.keras.preprocessing.image.img_to_array(img, dtype='float32')
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def load_image_from_buffer(image):
    image = PIL.Image.open(image)
    image = tf.keras.preprocessing.image.img_to_array(image, dtype='float32')
    image = np.expand_dims(image, axis=0)
    return image


def gram_matrix(x):
    """ Calculate gram matrix for given input tensor """
    # Calculate the summation part of the formula
    result = tf.einsum('bijc,bijd->bcd', x, x)
    # Divide summation part to (channels * height * width)
    # Given tensor's shape: (batch_size, height, width, channels)
    return result / (x.shape[1] * x.shape[2] * x.shape[3])


def remove_temp_images():
    for filename in os.listdir(settings.STATICFILE_IMAGES_DIR):
        if filename.startswith('temp_'):
            os.remove(os.path.join(settings.STATICFILE_IMAGES_DIR, filename))


def create_temp_img_path():
    filename = f'temp_{int(time.time())}.png'
    save_path = os.path.join(settings.STATICFILE_IMAGES_DIR, filename)
    return ['images/' + filename, save_path]