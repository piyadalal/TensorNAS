
from keras import backend as K

def channel_shuffle(input_shape, groups):
    """
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    groups: int
        number of groups per channel
    Returns
    -------
        channel shuffled output tensor
    """
    height, width, in_channels = input_shape.shape.as_list()[1:]
    channels_per_group = in_channels // groups

    x = K.reshape(input_shape, [-1, height, width, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
    x = K.reshape(x, [-1, height, width, in_channels])

    return x