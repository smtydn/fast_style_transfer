import numpy as np
import tensorflow as tf


def preprocess_image(image_path, target_size=None):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    img = np.expand_dims(img, axis=0)
    assert np.ndim(img) == 4

    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)
