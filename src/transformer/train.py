import os
import datetime
import functools

import tensorflow as tf
from tensorflow.keras.applications import vgg16
import numpy as np

from src import settings
from src.transformer import utils, models


class TransformNetTrainer:
    """ Trainer object for TransformNet
    Args:
        style_image_path: A `string` or `PathLike` object for style image path.
        content_image_path: A `string` or `PathLike` object for an image that
            will be used for sampling.
        train_dataset_path: A `string` or `PathLike` object for train dataset path.
        checkpoint_dir: A `string` or `PathLike` object for checkpoints.
        weights_dir: A `string` or `PathLike` object for weights.
        sample_dir: A `string` or `PathLike` object for samples.
        batch_size: An `int` value for batch size. Defaults to `4`.
        epochs: An `int` value for epochs. Defaults to `1`.
        learning_rate: An `int`. Learning rate for the optimizer. Defaults to `1e-3`.
        content_weight: A `float` value for content weight. Defaults to `1.0`.
        style_weight: A `float` value for style weight. Defaults to `1.0`.
        tv_weight: A `float` value for total variation loss weight. Defaults to `1e-6`.
        log_interval: An `int` for the interval of logging progress.
        checkpoint_interval: An `int` for the interval of saving current weights as checkpoint.
        sample_interval: An `int` for the interval of sampling. Defaults to `1000`.
        checkpoint_path: Optional. A `string` or `PathLike` object for checkpoint weights.
    """

    def __init__(
            self,
            style_image_path,
            content_image_path,
            train_dataset_path,
            checkpoint_dir,
            weights_dir,
            sample_dir,
            batch_size=4,
            epochs=1,
            learning_rate=1e-3,
            content_weight=1.0,
            style_weight=1.0,
            tv_weight=1e-6,
            log_interval=100,
            checkpoint_interval=2000,
            sample_interval=1000,
            checkpoint_path=None):

        self.epochs = epochs
        self.batch_size = batch_size
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.sample_interval = sample_interval
        self.weights_dir = weights_dir

        # Initialize LossNet.
        self.extractor = models.LossNet()

        # Initialize TransformNet. Load checkpoint if given.
        self.transformer = models.TransformNet()
        if checkpoint_path:
            self.transformer.build((1, 256, 256, 3))
            self.transformer.load_weights(checkpoint_path)

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Get style image name and gram matrix of style image features
        self.style_image_name = os.path.split(style_image_path)[1].split('.')[0]
        self.style_image_grams = self._get_style_grams(style_image_path)

        # Load content image
        self.content_image = utils.load_image_from_path(content_image_path)
        self.content_image_name = os.path.split(content_image_path)[1].split('.')[0]

        # Define training set
        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_dataset_path,
            label_mode=None,
            batch_size=batch_size
        )

    def _get_style_grams(self, image_path):
        style_image = utils.load_image_from_path(image_path)
        style_image_feats = self.extractor(style_image)
        return list(map(utils.gram_matrix, style_image_feats['style']))

    def _log_progress(self, epoch, batch_num, total_loss):
        print(f'Epoch: {epoch}\tBatch: {batch_num}\tLoss:{total_loss:.2f}\t\tTime:{datetime.datetime.now()}')

    def _save_checkpoint(self, epoch, batch_num):
        chkpt_path = os.path.join(self.checkpoint_dir, f'{self.style_image_name}-epoch_{epoch}-batch_{batch_num}.h5')
        self.transformer.save_weights(chkpt_path)
        print(f'Checkpoint saved. Path: {chkpt_path}')

    def _save_sample(self, epoch, batch_num):
        path = os.path.join(self.sample_dir, f'{self.style_image_name}-{self.content_image_name}-epoch_{epoch}-batch_{batch_num}.png')
        tf.keras.preprocessing.image.save_img(path, tf.squeeze(self.transformer(self.content_image)))
        print(f'Sample image has saved. Path: {path}')

    def _save_weights(self):
        weight_path = os.path.join(self.weights_dir, f'{self.style_image_name}.h5')
        self.transformer.save_weights(weight_path)
        print(f'Weights saved. Path: {weight_path}')

    def run(self):
        # Log start time
        start_time = datetime.datetime.now()
        print(f'Started at: {start_time}')

        # Training loop
        for epoch in range(1, self.epochs+1):
            batch_num = 1
            for batch in self.train_dataset:
                # Convert batch type to 'float32'
                batch = tf.cast(batch, 'float32')

                # Discard the last batch if does not have enough samples and stop
                if len(batch) != self.batch_size:
                    break

                with tf.GradientTape() as tape:
                    # Feed forward the batch, generate images
                    generated = self.transformer(batch)

                    # Get original image and generated image features
                    original_features = self.extractor(batch)
                    generated_features = self.extractor(generated)

                    # Extract generated image gram matrices from its features
                    generated_grams = map(utils.gram_matrix, generated_features['style'])

                    # Compute content loss
                    content_size = functools.reduce(lambda x, y: x * y, original_features['content'][0].shape)
                    content_loss = (self.content_weight / content_size) * tf.nn.l2_loss(
                        original_features['content'][0] - generated_features['content'][0]
                    )

                    # Compute style loss
                    style_loss = 0.0
                    for style_gram, generated_gram in zip(self.style_image_grams, generated_grams):
                        size = functools.reduce(lambda x, y: x * y, style_gram.shape)
                        style_loss += tf.nn.l2_loss(style_gram - generated_gram) / size
                    style_loss *= self.style_weight

                    # Compute total variation loss
                    tv_loss = self.tv_weight * tf.reduce_sum(tf.image.total_variation(generated))

                    # Compute total loss
                    total_loss = content_loss + style_loss + tv_loss

                # Apply gradient optimization for trainable transformer weights
                grads = tape.gradient(total_loss, self.transformer.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.transformer.trainable_weights))

                if batch_num % self.log_interval == 0:
                    self._log_progress(epoch, batch_num, total_loss)

                if batch_num % self.checkpoint_interval == 0:
                    self._save_checkpoint(epoch, batch_num)

                if batch_num % self.sample_interval == 0:
                    self._save_sample(epoch, batch_num)

                batch_num += 1
        
        # Save weights
        self._save_weights()

        # Log timers
        end_time = datetime.datetime.now()
        print(f'Finished at: {end_time}')
        print(f'Total time: {end_time - start_time}')

        print('Training completed.')        
