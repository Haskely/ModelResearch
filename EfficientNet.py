import copy
import math
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow as tf
layers.BatchNormalization
layers.LayerNormalization

DEFAULT_BLOCKS_ARGS = [
    {
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 32,
        'filters_out': 16,
        'expand_ratio': 1,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.25
    }, {
        'kernel_size': 3,
        'repeats': 2,
        'filters_in': 16,
        'filters_out': 24,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.25
    }, {
        'kernel_size': 5,
        'repeats': 2,
        'filters_in': 24,
        'filters_out': 40,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.25
    }, {
        'kernel_size': 3,
        'repeats': 3,
        'filters_in': 40,
        'filters_out': 80,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.25
    }, {
        'kernel_size': 5,
        'repeats': 3,
        'filters_in': 80,
        'filters_out': 112,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.25
    }, {
        'kernel_size': 5,
        'repeats': 4,
        'filters_in': 112,
        'filters_out': 192,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.25
    }, {
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 192,
        'filters_out': 320,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.25
    }
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Arguments:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.

    Returns:
      A tuple.
    """
    img_dim = 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def EfficientNet(
        width_coefficient,
        depth_coefficient,
        default_size,
        dropout_rate,
        input_shape=None,
        include_top=False,
        pooling=None
):
    width_coefficient = width_coefficient
    depth_coefficient = depth_coefficient
    # default_size = default_size
    dropout_rate = dropout_rate
    drop_connect_rate = 0.2
    depth_divisor = 8
    activation = 'swish'
    # blocks_args='default'
    model_name = 'efficientnet'
    include_top = include_top
    # weights = None
    # input_tensor = None
    pooling = pooling
    classes = 1000
    classifier_activation = 'softmax'

    blocks_args = DEFAULT_BLOCKS_ARGS

    # Determine proper input shape
    input_shape = input_shape if input_shape else (224, 224, 3)
    assert input_shape[0] >= default_size, input_shape[1] >= default_size

    img_input = layers.Input(shape=input_shape)

    bn_axis = 3

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(
            filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = img_input
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.)(x)
    # ?这个到底是什么Normalization呢？？？ 猜测是tensorflow.keras.layers.experimental.preprocessing.Normalization
    # >>> efnet = tf.keras.applications.EfficientNetB0()
    # >>> efnet.get_layer('rescaling')
    #  <tensorflow.python.keras.layers.preprocessing.image_preprocessing.Rescaling at 0x7fc774651f10>
    # >>> efnet.get_layer('normalization')
    #  <tensorflow.python.keras.layers.preprocessing.normalization.Normalization at 0x7fc774a94a60>
    # x = layers.Normalization(axis=bn_axis)(x) 根据文档，这个其实是需要手动在模型fit前adapt数据初始化参数的，我很好奇他为什么不像batch_normalization那样训练时自动adapt呢
    x = tf.keras.layers.experimental.preprocessing.Normalization(
        axis=bn_axis)(x)

    x = layers.ZeroPadding2D(
        padding=correct_pad(x, 3),
        name='stem_conv_pad')(x)
    x = layers.Conv2D(
        round_filters(32),
        3,
        strides=2,
        padding='valid',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)

    b = 0
    blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(
                x,
                activation,
                drop_connect_rate * b / blocks,
                name='block{}{}_'.format(i + 1, chr(j + 97)),
                **args)
            b += 1

    # Build top
    x = layers.Conv2D(
        round_filters(1280),
        1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)

        x = layers.Dense(
            classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name=model_name)

    return model


def block(inputs,
          activation='swish',
          drop_rate=0.,
          name='',
          filters_in=32,
          filters_out=16,
          kernel_size=3,
          strides=1,
          expand_ratio=1,
          se_ratio=0.,
          id_skip=True):
    """An inverted residual block.

    Arguments:
        inputs: input tensor.
        activation: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.

    Returns:
        output tensor for the block.
    """
    bn_axis = 3
    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'expand_conv')(
                inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = layers.Activation(activation, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=correct_pad(x, kernel_size),
            name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding='same',
            activation=activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'se_reduce')(
                se)
        se = layers.Conv2D(
            filters,
            1,
            padding='same',
            activation='sigmoid',
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'se_expand')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = layers.Conv2D(
        filters_out,
        1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')
    return x


EfficientNetB0_kwargs = {
    'width_coefficient': 1.0,
    'depth_coefficient': 1.0,
    'default_size': 224,
    'dropout_rate': 0.2,
}

EfficientNetB1_kwargs = {
    'width_coefficient': 1.0,
    'depth_coefficient': 1.1,
    'default_size': 240,
    'dropout_rate': 0.2,
}

EfficientNetB2_kwargs = {
    'width_coefficient': 1.1,
    'depth_coefficient': 1.2,
    'default_size': 260,
    'dropout_rate': 0.3,
}

EfficientNetB3_kwargs = {
    'width_coefficient': 1.2,
    'depth_coefficient': 1.4,
    'default_size': 300,
    'dropout_rate': 0.3,
}

EfficientNetB4_kwargs = {
    'width_coefficient':         1.4,
    'depth_coefficient':         1.8,
    'default_size':         380,
    'dropout_rate':         0.4
}

EfficientNetB5_kwargs = {
    'width_coefficient':        1.6,
    'depth_coefficient':        2.2,
    'default_size':        456,
    'dropout_rate':        0.4,
}

EfficientNetB6_kwargs = {
    'width_coefficient':         1.8,
    'depth_coefficient':         2.6,
    'default_size':         528,
    'dropout_rate':         0.5,
}

EfficientNetB7_kwargs = {
    'width_coefficient': 2.0,
    'depth_coefficient': 3.1,
    'default_size': 600,
    'dropout_rate': 0.5,
}

if __name__ == '__main__':
    from modelresearch import test_model
    test_model(
        EfficientNet(**EfficientNetB0_kwargs, input_shape=(1600, 256, 1)),
        'MyEfficientNetB0_ASR',
        )
