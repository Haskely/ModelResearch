import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


def MobileNetV3(input_shape=None, is_large=True, include_top=False, pooling=None):
    stack_fn = stack_fn_large if is_large else stack_fn_small
    last_point_ch = 1280 if is_large else 1024
    input_shape = input_shape if input_shape else (224, 224, 3)
    # alpha = 1.0
    model_type = 'large' if is_large else 'small'
    minimalistic = False
    include_top = include_top
    # weights = 'imagenet'
    # input_tensor = None
    classes = 1000
    pooling = pooling
    dropout_rate = 0.2
    classifier_activation = 'softmax'

    img_input = layers.Input(shape=input_shape)
    channel_axis = -1

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = img_input
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.)(x)
    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv/BatchNorm')(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(K.int_shape(x)[channel_axis] * 6)

    x = layers.Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv_1/BatchNorm')(x)
    x = activation(x)
    x = layers.Conv2D(
        last_point_ch,
        kernel_size=1,
        padding='same',
        use_bias=True,
        name='Conv_2')(x)
    x = activation(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Reshape((1, 1, last_point_ch))(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(classes, kernel_size=1,
                          padding='same', name='Logits')(x)
        x = layers.Flatten()(x)
        x = layers.Activation(activation=classifier_activation,
                              name='Predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name='MobilenetV3' + model_type)

    return model


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


def stack_fn_small(x, kernel, activation, se_ratio):

    x = _inverted_res_block(x, 1, 16, 3, 2, se_ratio, relu, 0)
    x = _inverted_res_block(x, 72. / 16, 24, 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 88. / 24, 24, 3, 1, None, relu, 2)
    x = _inverted_res_block(x, 4, 40, kernel, 2, se_ratio, activation, 3)
    x = _inverted_res_block(x, 6, 40, kernel, 1, se_ratio, activation, 4)
    x = _inverted_res_block(x, 6, 40, kernel, 1, se_ratio, activation, 5)
    x = _inverted_res_block(x, 3, 48, kernel, 1, se_ratio, activation, 6)
    x = _inverted_res_block(x, 3, 48, kernel, 1, se_ratio, activation, 7)
    x = _inverted_res_block(x, 6, 96, kernel, 2, se_ratio, activation, 8)
    x = _inverted_res_block(x, 6, 96, kernel, 1, se_ratio, activation, 9)
    x = _inverted_res_block(x, 6, 96, kernel, 1, se_ratio, activation, 10)
    return x


def stack_fn_large(x, kernel, activation, se_ratio):

    x = _inverted_res_block(x, expansion=1, filters=16, kernel_size=3,
                            stride=1, se_ratio=None, activation=relu, block_id=0)
    x = _inverted_res_block(x, 4, 24, 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 3, 24, 3, 1, None, relu, 2)
    x = _inverted_res_block(x, 3, 40, kernel, 2, se_ratio, relu, 3)
    x = _inverted_res_block(x, 3, 40, kernel, 1, se_ratio, relu, 4)
    x = _inverted_res_block(x, 3, 40, kernel, 1, se_ratio, relu, 5)
    x = _inverted_res_block(x, 6, 80, 3, 2, None, activation, 6)
    x = _inverted_res_block(x, 2.5, 80, 3, 1, None, activation, 7)
    x = _inverted_res_block(x, 2.3, 80, 3, 1, None, activation, 8)
    x = _inverted_res_block(x, 2.3, 80, 3, 1, None, activation, 9)
    x = _inverted_res_block(x, 6, 112, 3, 1, se_ratio, activation, 10)
    x = _inverted_res_block(x, 6, 112, 3, 1, se_ratio, activation, 11)

    x = _inverted_res_block(x, 6, 160, kernel, 2, se_ratio, activation, 12)
    x = _inverted_res_block(x, 6, 160, kernel, 1, se_ratio, activation, 13)
    x = _inverted_res_block(x, 6, 160, kernel, 1, se_ratio, activation, 14)
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                        activation, block_id):
    channel_axis = -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = K.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'expand')(
                x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand/BatchNorm')(
                x)
        x = activation(x)

    if stride == 2:

        input_size = K.int_shape(x)[1:3]
        kernel_size = (kernel_size, kernel_size)
        if input_size[0] is None:
            adjust = (1, 1)
        else:
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
        correct = (kernel_size[0] // 2, kernel_size[1] // 2)
        padding = ((correct[0] - adjust[0], correct[0]),
                   (correct[1] - adjust[1], correct[1]))

        x = layers.ZeroPadding2D(
            padding=padding,
            name=prefix + 'depthwise/pad')(
                x)

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'depthwise')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise/BatchNorm')(
            x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'project')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project/BatchNorm')(
            x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(
        inputs)
    x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv')(
            x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv_1')(
            x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _depth(v):
    divisor = 8
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


if __name__ == '__main__':
    from modelresearch import test_model
    is_large = True

    test_model(MobileNetV3((1600, 256, 1), is_large=is_large),
               f"MyMobileNetV3{'Large' if is_large else 'Small'}_ASR")
