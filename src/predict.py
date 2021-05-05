import os

import tensorflow as tf

import settings
from src import utils
from src.models import TransformNet


class StyleTransformer:
    
    def __init__(self, style_name):
        self.style_name = style_name
        self.weights_path = settings.AVAILABLE_STYLES[self.style_name]
        self.model = self._load_model()

    def _load_model(self):
        """ Load TransformNet weights to be able to predict. """
        model = TransformNet()
        model(tf.ones(shape=(1, 256, 256, 3)))  # Give a dummy input to make model be able to load
        model.load_weights(self.weights_path)
        return model

    def predict(self, image_path):
        """ Main function. Makes prediction and saves image."""
        if os.path.exists(settings.GENERATED_IMG_PATH):
            os.remove(settings.GENERATED_IMG_PATH)

        image = utils.load_image(image_path)
        result = self.model(image)
        result = tf.squeeze(result, axis=0)     # Remove batch dimension
        tf.keras.preprocessing.image.save_img(settings.GENERATED_IMG_PATH, result)



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