def rpn_layer(base_layers, num_anchors):
    """
    Args:
        base_layers: vgg in here
        num_anchors: 9 in here
    Returns:
        List - [x_class, x_regr, base_layers]
        here,
        x_class     : Classification (whether it's an object or not)
        x_regr      : Bounding Box regression
        base_layers : VGG-16
    """
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)

    x_regr  = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]