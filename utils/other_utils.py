import torch
import numpy as np


def squash(input_tensor, dim=-1, epsilon=1e-7):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector


def coord_addition(input_tensor, shuffle=False):
    """
    adds the coordinates to the input tensor
    :param input_tensor: tensor of shape [batch_size, capsule_dim, num_capsule_maps, H, W]
    :return: tensor of shape [batch_size, capsule_dim+2, num_capsule_maps, H, W]
    """
    batch_size, _, num_maps, H, W = input_tensor.size()
    w_offset_vals = np.tile(np.reshape((np.arange(W) + 0.50) / float(W), (1, -1)), (H, 1))
    h_offset_vals = np.tile(np.reshape((np.arange(H) + 0.50) / float(H), (-1, 1)), (1, W))
    coordinates = np.stack([w_offset_vals] + [h_offset_vals], axis=0)   # [2, H, W]
    if not shuffle:
        coordinates = torch.tensor(coordinates[None, :, None, :, :]).repeat(batch_size, 1, num_maps, 1, 1)
    else:
        coords = np.repeat(coordinates[None, :, None, :, :], batch_size, axis=0)
        for i in range(num_maps):
            pass


    # shuffle
    # np.random.shuffle(np.reshape(coordinates, -1))


    # coordinates = np.zeros_like(coordinates)

    out_tensor = torch.cat((input_tensor, coordinates.float().cuda()), 1)
    return out_tensor

