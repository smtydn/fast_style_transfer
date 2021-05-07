import os
import io
import base64

import PIL
import tensorflow as tf

from src import settings
from src.transformer import utils
from src.transformer.models import TransformNet


class StyleTransformer:
    """ Prediction implementation for TransformNet. It both supports buffer output and saving. """
    
    def __init__(self, style_name, weights_path=None, buffer=None):
        self.style_name = style_name

        # For CLI implementation of prediction
        # Get weights from a path if provided
        self.weights_path = weights_path if weights_path else settings.AVAILABLE_STYLES[self.style_name]
        self.model = self._load_model()
        self.buffer = buffer if buffer else io.BytesIO()

    def _load_model(self):
        """ Load TransformNet weights to be able to predict. """
        model = TransformNet()
        model(tf.ones(shape=(1, 256, 256, 3)))  # Give a dummy input to make model be able to load
        model.load_weights(self.weights_path)
        return model

    def _deprocess_image(self, model_output, save_path=None, return_decoded=True):
        """ Deprocess image. save=False to make it ready for serving with flask. Else save image to given path."""
        result = tf.squeeze(model_output, axis=0)     # Remove batch dimension
        if save_path:
            tf.keras.preprocessing.image.save_img(save_path, result)
        if return_decoded:
            result = tf.convert_to_tensor(result)   # EagerTensor -> Tensor. Needed for array_to_img
            image = tf.keras.preprocessing.image.array_to_img(result)
            image.save(self.buffer, format='PNG')   # Save image to buffer
            img_encoded = base64.b64encode(self.buffer.getvalue())  # bytes-like -> bytes
            return 'data:image/png;base64, ' + img_encoded.decode('utf-8')  # Return decoded version            

    def predict(self, image=None, image_path=None, save_path=None, return_decoded=True):
        """ Main function. Makes prediction and saves image to buffer."""
        if image_path:
            image = utils.load_image_from_path(image_path)
        elif image:
            image = utils.load_image_from_buffer(image)
        else:
            raise Exception("'image' or 'image_path' must be provided.")

        result = self.model(image)
        return self._deprocess_image(result, save_path=save_path, return_decoded=return_decoded)
