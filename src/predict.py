import keras
import tensorflow as tf

from src import utils
from src.models import TransformNet


def predict():
    weights_path = r'C:\Users\samet\Projects\fast_style_transfer\weights\checkpoints\la_muse.jpg-epoch1-batch7000.h5'
    content_path = r'C:\Users\samet\Projects\fast_style_transfer\images\content\zurich.jpeg'
    output_path = r'C:\Users\samet\Projects\fast_style_transfer\images\output\zurich.jpg'


    model = TransformNet()
    model(tf.ones(shape=(1, 256, 256, 3)))
    model.load_weights(weights_path)

    content_image = utils.preprocess_image(content_path)
    res = model(content_image)
    res = tf.squeeze(res)

    keras.preprocessing.image.save_img(output_path, res)
    print('Image has saved.')