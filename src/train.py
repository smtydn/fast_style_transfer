import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import os
import datetime
import functools

import tensorflow as tf
from keras.applications import vgg16

import settings
from src import utils
from src.models import TransformNet, LossNet


def start(learning_rate, style_image_name, batch_size, image_size, content_weight, style_weight,
            log_interval, chkpt_interval, epochs, sample_interval, content_image):
    # Initialize networks
    loss_net = LossNet()
    style_net = TransformNet()

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Load style image
    style_image_path = os.path.join(settings.STYLE_IMAGES_DIR, style_image_name)
    style_image_name = style_image_name.split('.')[0]
    style_image = utils.preprocess_image(style_image_path)

    # Extract style features
    style_features = loss_net.predict(style_image)
    style_gram = map(utils.gram_matrix, style_features['style'])

    # Define training dataset loader
    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_dataset = datagen.flow_from_directory(
        settings.TRAIN_DATASET_PATH, 
        target_size=image_size,
        class_mode=None,
        batch_size=batch_size
    )

    # Log start time
    start_time = datetime.datetime.now()
    print(f'Started at: {start_time}')

    # Define training loop
    for epoch in range(1, epochs+1):
        print(f'Epoch: {epoch}')
        batch_num = 1
        for batch in train_dataset:
            # Discard the last batch if does not have enough samples
            if len(batch) != batch_size:
                break

            batch = tf.image.per_image_standardization(batch)
            with tf.GradientTape() as tape:
                # Feed forward the batch, generate images
                generated = style_net(batch)

                # Get original image and generated image features
                orig_feats = loss_net(batch)
                generated_feats = loss_net(generated)

                # Compute content loss
                content_size = functools.reduce(lambda x, y: x * y, orig_feats['content'][0].shape)
                content_loss = content_weight * tf.nn.l2_loss(
                    orig_feats['content'][0] - generated_feats['content'][0]
                ) / content_size

                # Compute style loss
                style_loss = 0.0
                generated_gram = map(utils.gram_matrix, generated_feats['style'])
                for gr_style, gr_gen in zip(style_gram, generated_gram):
                    size = functools.reduce(lambda x, y: x * y, gr_style.shape)
                    style_loss += tf.nn.l2_loss(gr_style - gr_gen) / size
                style_loss *= style_weight

                # Compute total loss
                total_loss = content_loss + style_loss
            
            grads = tape.gradient(total_loss, style_net.trainable_weights)
            optimizer.apply_gradients(zip(grads, style_net.trainable_weights))

            if batch_num % log_interval == 0:
                print(f'Batch: {batch_num}\tLoss:{total_loss:.2f}\t\tTime:{datetime.datetime.now()}')

            if batch_num % chkpt_interval == 0:
                chkpt_path = os.path.join(settings.CHECKPOINTS_DIR, f'{style_image_name}-epoch{epoch}-batch{batch_num}.h5')
                style_net.save_weights(chkpt_path)
                print(f'Checkpoint saved. Path: {chkpt_path}')

            if batch_num % sample_interval == 0:
                preprocessed_sample = utils.preprocess_image(os.path.join(settings.CONTENT_IMAGES_DIR, content_image))
                preprocessed_sample *= 1.0 / 255.0
                tf.keras.preprocessing.image.save_img(
                    os.path.join(settings.OUTPUT_IMAGES_DIR, f'epoch{epoch}-batch{batch_num}-{content_image}'), 
                    tf.squeeze(style_net(preprocessed_sample))
                )
                print('Sample image has saved.')

            batch_num += 1

    # Save weights
    weight_path = os.path.join(settings.WEIGHTS_DIR, f'{style_image_name}.h5')
    style_net.save_weights(weight_path)
    print(f'Weights saved. Path: {weight_path}')

    # Log end time
    end_time = time.time()
    print(f'Finished at: {datetime.datetime.now()}')
    print(f'Total time: {end_time - start_time}')
    print('Training completed.')