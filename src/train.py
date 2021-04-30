import os
import tensorflow as tf

import utils
from models import TransformNet, LossNet


# define constants
style_image_path = r'C:\Users\samet\Projects\fast_style_transfer\images\style\the-scream.jpg'
train_dataset_path = r'D:\Datasets\COCO2014'
weight_save_path = r'C:\Users\samet\Projects\fast_style_transfer\weights'
batch_size = 4
image_size = (256, 256)
learning_rate = 1e-3
epochs = 1
style_weight = 1e-2
content_weight = 1e4
loss_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
content_layer = 'block3_conv3'

# define networks
loss_net = LossNet()
style_net = TransformNet()

# define optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# load style image
style_image = utils.preprocess_image(style_image_path)
assert style_image.shape == (1, 720, 565, 3)

# compute style features
style_features = loss_net(style_image)
# style_gram = [utils.gram_matrix(output) for layer_name, output in sorted(style_features.items())]

# define train dataset loader
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dataset_path,
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    label_mode=None
)

# define training loop
for epoch in range(epochs):
    batch_num = 1
    for batch in train_dataset:
        # Discard the last batch if does not have enough samples
        if len(batch) != batch_size:
            break
        
        with tf.GradientTape() as tape:
            # Feed forward the batch, generate an image
            generated_images = style_net(batch)

            # Get original image and generated image features
            orig_features = loss_net(batch)
            generated_features = loss_net(generated_images)

            # Compute content loss
            content_loss = content_weight * tf.math.reduce_sum(tf.keras.losses.MSE(orig_features[content_layer], generated_features[content_layer]))

            # Compute style loss
            style_loss = 0.0
            for layer_name, layer_output in generated_features.items():
                style_gram = utils.gram_matrix(style_features[layer_name])
                generated_gram = utils.gram_matrix(layer_output)
                style_loss += tf.math.reduce_sum(tf.keras.losses.MSE(style_gram, generated_gram))
            style_loss *= style_weight

            # Compute total loss
            total_loss = content_loss + style_loss
        
        grad = tape.gradient(total_loss, style_net.trainable_weights)
        optimizer.apply_gradients(zip(grad, style_net.trainable_weights))

        if batch_num % 50 == 0:
            print(f'Batch: {batch_num}\tLoss:{total_loss:.2f}')

        if batch_num % 1000 == 0:
            style_net.save_weights(os.path.join(weight_save_path, f'the-scream-batch{batch_num}.h5'))
            break

        batch_num += 1