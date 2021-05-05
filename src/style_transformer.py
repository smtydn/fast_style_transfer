import os
import io
import base64

import tensorflow as tf

import settings
from src import utils
from src.models import TransformNet


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

    def _deprocess_image(self, model_output, save=False, save_path=None):
        """ Deprocess image. save=False to make it ready for serving with flask. Else save image to given path."""
        result = tf.squeeze(model_output, axis=0)     # Remove batch dimension
        if not save:
            result = tf.convert_to_tensor(result)   # EagerTensor -> Tensor. Needed for array_to_img
            image = tf.keras.preprocessing.image.array_to_img(result)
            image.save(self.buffer, format='PNG')   # Save image to buffer
            img_encoded = base64.b64encode(self.buffer.getvalue())  # bytes-like -> bytes
            return img_encoded.decode('utf-8')  # Return decoded version
        else:
            tf.keras.preprocessing.image.save_img(save_path, result)


    def predict(self, image_path, save=False, save_path=None):
        """ Main function. Makes prediction and saves image to buffer."""
        image = utils.load_image(image_path)
        result = self.model(image)
        return self._deprocess_image(result, save=save, save_path=save_path)


# def predict(style_name, image_path, output_path):
#     weights_path = os.path.join(settings.WEIGHTS_DIR, f'{style_name}.h5')

#     model = TransformNet()
#     model(tf.ones(shape=(1, 256, 256, 3)))
#     model.load_weights(weights_path)

#     content_image = utils.load_image(image_path)
#     res = model(content_image)
#     res = tf.squeeze(res)

#     tf.keras.preprocessing.image.save_img(output_path, res)
#     print('Image has saved.')