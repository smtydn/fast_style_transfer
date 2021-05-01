import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16


def preprocess_image(image_path, target_size=None, vgg_preprocess=False):
    # Load image from the path, resize if it needed
    img = load_img(image_path, target_size=target_size)
    # Convert image to a Numpy array
    img = img_to_array(img)
    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    if vgg_preprocess:
        # Apply VGG16 preprocessing
        img = vgg16.preprocess_input(img)
    return img

def gram_matrix(x):
    # Calculate the summation part of the formula
    result = tf.einsum('bijc,bijd->bcd', x, x)
    # Divide summation part to (channels * height * width)
    # Given tensor's shape: (batch_size, height, width, channels)
    return result / (x.shape[1] * x.shape[2] * x.shape[3])