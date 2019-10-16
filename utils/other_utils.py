import torch


def one_hot(y, num_cls):
    m = torch.eye(num_cls).cuda()
    return m.index_select(dim=0, index=y)



