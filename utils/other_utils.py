import torch


def one_hot(y, num_cls):
    m = torch.eye(num_cls).cuda()
    return m.index_select(dim=0, index=y)


def squash(input_tensor, dim=-1, epsilon=1e-7):
    squared_norm = (input_tensor ** 2).sum(dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = input_tensor / safe_norm
    return squash_factor * unit_vector


