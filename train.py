import time
import os

import tensorflow as tf
import numpy as np
import keras
from keras.applications import vgg16

import models
import utils

tf.config.run_functions_eagerly(True)
AUTOTUNE = tf.data.AUTOTUNE

WIDTH = 256
HEIGHT = 256
STYLE_IMAGE_PATH = 'C:\\Users\\samet\\Projects\\fast_style_transfer\\images\\style\\the_scream.jpg'
STYLE_IMAGE_NAME = os.path.split(STYLE_IMAGE_PATH)[-1].split('.')[0]
STYLE_LAYERS = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"]
CONTENT_LAYER = "block2_conv2"
BATCH_SIZE = 4
STYLE_WEIGHT = 1e2
CONTENT_WEIGHT = 7.5
TV_WEIGHT = 2e2
LEARNING_RATE = 1e-3 
EPOCHS = 1
TRAIN_DATASET_PATH = "D:\\Datasets\\test"
WEIGHT_SAVE_PATH = 'C:\\Users\\samet\\Projects\\fast_style_transfer\\weights'

# define dummy loss
dummy_loss = lambda y_true, y_pred: y_pred

# precompute style layer activations
style_image = utils.preprocess_image(STYLE_IMAGE_PATH, WIDTH, HEIGHT)
style_acts = []
for layer_name in STYLE_LAYERS:
    fn = utils.get_vgg_activation(layer_name, WIDTH, HEIGHT)
    style_acts.append(utils.expand_input(BATCH_SIZE, fn([style_image])[0]))

# build training model
model = models.training_model(bs=BATCH_SIZE)
model.compile(
    loss={'content': dummy_loss, 'style1': dummy_loss, 'style2': dummy_loss, 'style3': dummy_loss, 'style4': dummy_loss, 'tv': dummy_loss, 'output': dummy_loss},
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss_weights=[CONTENT_WEIGHT, STYLE_WEIGHT, STYLE_WEIGHT, STYLE_WEIGHT, STYLE_WEIGHT, TV_WEIGHT, 0]    
)


# prepare for training
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=TRAIN_DATASET_PATH,
    batch_size=BATCH_SIZE,
    image_size=(HEIGHT, WIDTH),
    shuffle=True,
    label_mode=None
).prefetch(buffer_size=AUTOTUNE)
dummy_in = utils.expand_input(BATCH_SIZE, np.array([0.0]))
c_loss = None

for e in range(1, EPOCHS + 1):
    print("Epoch: %d" % (e))

    for batch_num, batch in enumerate(train_dataset):
        t1 = time.time()
        x = vgg16.preprocess_input(batch)
        content_act = utils.get_vgg_activation(CONTENT_LAYER, WIDTH, HEIGHT)([x])[0]

        res = model.fit(
            [x, content_act, style_acts[0], style_acts[1], style_acts[2], style_acts[3]],
            [dummy_in, dummy_in, dummy_in, dummy_in, dummy_in, dummy_in, x],
            epochs=1, verbose=0, batch_size=BATCH_SIZE
        )

        if batch_num % 1 == 0:
            loss = res.history['loss'][0]
            print('Batch: %d, Loss: %.0f' % (batch_num, loss))
        if batch_num % 100 == 0:
            break

# Saving models
print("Saving models...")
model_eval = models.image_transform_network(WIDTH, HEIGHT)
training_model_layers = {layer.name: layer for layer in model.layers}
for layer in model_eval.layers:
    if layer.name in training_model_layers:
        layer.set_weights(training_model_layers[layer.name].get_weights())

model_eval.save_weights(os.path.join(WEIGHT_SAVE_PATH, f'{STYLE_IMAGE_NAME}.h5'))
print('Model has saved!')