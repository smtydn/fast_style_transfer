import os
import io
import base64

import PIL
import tensorflow as tf

from src import settings
from src.transformer import utils
from src.transformer.models import TransformNet


class StyleTransformer:
    """ Prediction implementation for TransformNet.
    Args:
        weights_path: The `string` or `PathLike` object for TransformNet weights.

    Once the object is initialized with given `weights_path`,
    prediction will be available with the method `predict()`.
    """
    
    def __init__(self, weights_path):
        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path):
        """ Load TransformNet weights. 
        Returns:
            TransformNet model with weights loaded.
        """
        model = TransformNet()
        model.build((1, 256, 256, 3))
        model.load_weights(weights_path)
        return model

    def _deprocess_image(self, model_output, save_path):
        """ Remove batch dimension. Save image. 
        Args:
            model_output: A `Tensor`. Expected to be the output from TransformNet.
            save_path: A path `string` or `PathLike` object. Defines the path
                that the image will be saved.
        """
        result = tf.squeeze(model_output, axis=0)     # Remove batch dimension
        tf.keras.preprocessing.image.save_img(save_path, result)

    def predict(self, save_path, image=None, image_path=None):
        """ Predict the output of TransformNet for given style image.
        Args:
            save_path: A path `string` or `PathLike` object. Defines the path
                that the image will be saved.
            image: A `FileLike` object. Expected to be the image provided to TransformerNet
                to predict. Designed for Web API.
            image_path: A path `string` or `PathLike` object. Defines the path
                that the content image is located.

        Raises:
            AttributeError: Only one of the parameters `image` or `image_path`
                should be provided.
        """
        if image and image_path:
            raise AttributeError('Only one of the `image` or `image_path` should be provided.')

        if image_path:
            img = utils.load_image_from_path(image_path)
        elif image:
            img = utils.load_image_from_file(image)
        else:
            raise AttributeError('At least one of the `image` or `image_path` should be provided.')

        result = self.model(img)
        return self._deprocess_image(result, save_path)
