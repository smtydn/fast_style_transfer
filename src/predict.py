import os
import tensorflow as tf

import settings
from src import utils
from src.models import TransformNet


def predict(style_name, image_path, output_path):
    weights_path = os.path.join(settings.WEIGHTS_DIR, f'{style_name}.h5')

    model = TransformNet()
    model(tf.ones(shape=(1, 256, 256, 3)))
    model.load_weights(weights_path)

    content_image = utils.load_image(image_path)
    res = model(content_image)
    res = tf.squeeze(res)

    tf.keras.preprocessing.image.save_img(output_path, res)
    print('Image has saved.')