import os
import datetime
import functools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras.applications import vgg16
import numpy as np

from src import settings
from src.transformer import utils, models


def start(learning_rate, style_image_name, batch_size, image_size, content_weight, style_weight,
            log_interval, chkpt_interval, epochs, sample_interval, content_image, tv_weight, chkpt_path):
    # Initialize networks
    extractor = models.LossNet()
    transformer = models.TransformNet()

    # Load checkpoint if given
    if chkpt_path:
        transformer.build(input_shape=(1, 256, 256, 3))
        transformer.load_weights(chkpt_path)

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Load style image
    style_image_path = os.path.join(settings.STYLE_IMAGES_DIR, style_image_name)
    style_image_name = style_image_name.split('.')[0]
    style_image = utils.load_image(style_image_path)

    # Extract style features
    style_features = extractor(style_image)
    style_grams = list(map(utils.gram_matrix, style_features['style']))

    # Define training dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        settings.TRAIN_DATASET_PATH,
        label_mode=None, 
        batch_size=batch_size,
        image_size=image_size,
        interpolation='nearest'
    )

    # Log start time
    start_time = datetime.datetime.now()
    print(f'Started at: {start_time}')

    # Define training loop
    for epoch in range(1, epochs+1):
        print(f'Epoch: {epoch}')
        batch_num = 1
        for batch in train_dataset:
            # Convert batch type to 'float32'
            batch = tf.cast(batch, dtype='float32')

            # Discard the last batch if does not have enough samples
            if len(batch) != batch_size:
                break
            
            with tf.GradientTape() as tape:
                # Feed forward the batch, generate images
                transformed = transformer(batch)

                # Get original image and transformed image features
                original_feats = extractor(batch)
                transformed_feats = extractor(transformed)

                # Compute content loss
                content_size = functools.reduce(lambda x, y: x * y, original_feats['content'][0].shape)
                content_loss = content_weight * tf.nn.l2_loss(
                    original_feats['content'][0] - transformed_feats['content'][0]
                ) / content_size

                # Compute style loss
                style_loss = 0.0
                transformed_grams = map(utils.gram_matrix, transformed_feats['style'])
                for style_gram, transformed_gram in zip(style_grams, transformed_grams):
                    size = functools.reduce(lambda x, y: x * y, style_gram.shape)
                    style_loss += tf.nn.l2_loss(style_gram - transformed_gram) / size
                style_loss *= style_weight

                # Compute total variation loss
                tv_loss = tv_weight * tf.reduce_sum(tf.image.total_variation(transformed))

                # Compute total loss
                total_loss = content_loss + style_loss + tv_loss
            
            grads = tape.gradient(total_loss, transformer.trainable_weights)
            optimizer.apply_gradients(zip(grads, transformer.trainable_weights))

            if batch_num % log_interval == 0:
                print(f'Batch: {batch_num}\tLoss:{total_loss:.2f}\t\tTime:{datetime.datetime.now()}')

            if batch_num % chkpt_interval == 0:
                chkpt_path = os.path.join(settings.CHECKPOINTS_DIR, f'{style_image_name}-epoch{epoch}-batch{batch_num}.h5')
                transformer.save_weights(chkpt_path)
                print(f'Checkpoint saved. Path: {chkpt_path}')

            if batch_num % sample_interval == 0:
                preprocessed_sample = utils.load_image(os.path.join(settings.CONTENT_IMAGES_DIR, content_image))
                tf.keras.preprocessing.image.save_img(
                    os.path.join(settings.SAMPLE_IMAGES_DIR, f'epoch{epoch}-batch{batch_num}-{style_image_name}-{content_image}'), 
                    tf.squeeze(transformer(preprocessed_sample))
                )
                print('Sample image has saved.')

            batch_num += 1

    # Save weights
    weight_path = os.path.join(settings.WEIGHTS_DIR, f'{style_image_name}.h5')
    transformer.save_weights(weight_path)
    print(f'Weights saved. Path: {weight_path}')

    # Log end time
    end_time = datetime.datetime.now()
    print(f'Finished at: {end_time}')
    print(f'Total time: {end_time - start_time}')
    print('Training completed.')