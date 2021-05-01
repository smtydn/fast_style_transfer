import os
import datetime

import tensorflow as tf
from keras.applications import vgg16

import settings
from src import utils
from src.models import TransformNet, LossNet


def start(learning_rate, style_image_name, batch_size, image_size, content_weight, style_weight,
            log_interval, chkpt_interval, epochs):
    # Log start time
    start_time = datetime.datetime.now()
    print(f'Started at: {start_time}')

    # Initialize networks
    loss_net = LossNet()
    style_net = TransformNet()

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Load style image, compute its features
    style_image = utils.preprocess_image(
        os.path.join(settings.STYLE_IMAGES_DIR, f'{style_image_name}.jpg'), vgg_preprocess=True
    )
    style_features = loss_net(style_image)

    # Define training dataset loader
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        settings.TRAIN_DATASET_PATH,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        label_mode=None
    )

    # Define training loop
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')
        batch_num = 1
        for batch in train_dataset:
            # Discard the last batch if does not have enough samples
            if len(batch) != batch_size:
                break
            
            with tf.GradientTape() as tape:
                # Feed forward the batch, generate images
                generated_images = style_net(batch)

                # Get original image and generated image features
                original_image_features = loss_net(vgg16.preprocess_input(batch))
                generated_image_features = loss_net(vgg16.preprocess_input(generated_images))

                # Compute content loss
                content_loss = content_weight * tf.math.reduce_sum(
                    tf.keras.losses.MSE(original_image_features['content'], generated_image_features['content'])
                )

                # Compute style loss
                style_loss = 0.0
                for orig_feat, gen_feat in zip(original_image_features['style'], generated_image_features['style']):
                    style_loss += tf.math.reduce_sum(tf.keras.losses.MSE(
                        utils.gram_matrix(orig_feat),
                        utils.gram_matrix(gen_feat)
                    ))
                style_loss *= style_weight

                # Compute total loss
                total_loss = content_loss + style_loss
            
            grads = tape.gradient(total_loss, style_net.trainable_weights)
            optimizer.apply_gradients(zip(grads, style_net.trainable_weights))

            if batch_num % log_interval == 0:
                print(f'Batch: {batch_num}\tLoss:{total_loss:.2f}\tTime:{time.time()}')

            if batch_num % chkpt_interval == 0:
                chkpt_path = os.path.join(settings.CHECKPOINTS_DIR, f'{style_image_name}-epoch{epoch+1}-batch{batch_num}.h5')
                style_net.save_weights(chkpt_path)
                print(f'Checkpoint saved. Path: {chkpt_path}')

            batch_num += 1

    weight_path = os.path.join(settings.WEIGHTS_DIR, f'{style_image_name}.h5')
    style_net.save_weights(weight_path)
    print(f'Weights saved. Path: {weight_path}')

    end_time = time.time()
    print(f'Finished at: {datetime.datetime.now()}')
    print(f'Total time: {end_time - start_time}')
    print('Training completed.')