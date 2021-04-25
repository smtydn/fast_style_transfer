from keras import backend as K


def get_content_loss(new_activation, content_activation):
    shape = K.cast(K.shape(content_activation), dtype='float32')
    return K.sum(K.square(new_activation - content_activation)) / (shape[1] * shape[2] * shape[3])


def gram_matrix(activation):
    assert K.ndim(activation) == 3
    shape = K.cast(K.shape(activation), dtype='float32')
    shape = (shape[0] * shape[1], shape[2])

    activation = K.reshape(activation, shape)
    return K.dot(K.transpose(activation), activation) / (shape[0] * shape[1])


def get_style_loss(batch_size, new_activation, style_activation):
    loss_sum = K.variable(0.0)
    for i in range(batch_size):
        ori_gram_matrix = gram_matrix(style_activation[i])
        new_gram_matrix = gram_matrix(new_activation[i])
        loss_sum = loss_sum + K.sum(K.square(ori_gram_matrix - new_gram_matrix))
    return loss_sum


def get_total_var_loss(args, **kwargs):
    image = args[0]
    width = kwargs['width']
    height = kwargs['height']
    x_diff = K.square(image[:, :height - 1, :, :] - image[:, 1:, :, :])
    y_diff = K.square(image[:, :, :width - 1, :] - image[:, :, 1:, :])
    x_diff = K.sum(x_diff) / height
    y_diff = K.sum(y_diff) / width
    return x_diff + y_diff