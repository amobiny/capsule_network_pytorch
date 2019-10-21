import numpy as np
import torch


def add_coord(input_tensor):
    """

    :param input_tensor: input tensor of shape [batch_size, primary_capsule_dim, num_primary_caps_map, height, width]
    :return:
    """

    height = input_tensor.shape[-1]
    width = input_tensor.shape[-2]
    dims = input_tensor.shape[-1]
    # height = input_tensor.shape[-1]
    # width = input_tensor.shape[-2]
    # dims = input_tensor.shape[-1]

    # Generate offset coordinates
    # The difference here is that the coordinate won't be exactly in the middle of
    # the receptive field, but will be evenly spread out
    w_offset_vals = (np.arange(width) + 0.50) / float(width)
    h_offset_vals = (np.arange(height) + 0.50) / float(height)

    w_offset = np.zeros([width, dims])  # (5, 16)
    w_offset[:, 3] = w_offset_vals
    # (1, 1, 5, 1, 1, 16)
    w_offset = np.reshape(w_offset, [1, 1, width, 1, 1, dims])

    h_offset = np.zeros([height, dims])
    h_offset[:, 7] = h_offset_vals
    # (1, 5, 1, 1, 1, 16)
    h_offset = np.reshape(h_offset, [1, height, 1, 1, 1, dims])

    # Combine w and h offsets using broadcasting
    # w is (1, 1, 5, 1, 1, 16)
    # h is (1, 5, 1, 1, 1, 16)
    # together (1, 5, 5, 1, 1, 16)
    offset = w_offset + h_offset

    # Convent from numpy to tensor
    # offset = torch.tensor(offset)
    input_tensor += offset

    # offset = tf.constant(offset, dtype=tf.float32)
    # votes = tf.add(votes, offset, name="votes_with_coord_add")
    return input_tensor


input_tensor = np.zeros((64, 5, 5, 32, 5, 16))
add_coord(input_tensor)
# input_tensor = np.zeros((100, 8, 32, 6, 6))
