import keras

import utils
import models


def predict(img_read_path, img_write_path):
    # Read image
    content = utils.preprocess_image(img_read_path, -1, -1, resize=False)
    ori_height = content.shape[1]
    ori_width = content.shape[2]

    # Pad image
    content = utils.get_padding(content)
    height = content.shape[1]
    width = content.shape[2]

    # Get eval model
    eval_model = models.image_transform_network(height, width)
    eval_model.load_weights('C:\\Users\\samet\\Projects\\fast_style_transfer\\weights\\the_scream.h5')

    # Generate output and save image
    res = eval_model.predict([content])
    output = utils.deprocess_image(res[0], width, height)
    output = utils.remove_padding(output, ori_height, ori_width)

    keras.preprocessing.image.save_img(img_write_path, output)



predict(
    'C:\\Users\\samet\\Projects\\fast_style_transfer\\images\\content\\chicago.jpg',
    'C:\\Users\\samet\\Projects\\fast_style_transfer\\images\\output\\chicago.jpg'
)