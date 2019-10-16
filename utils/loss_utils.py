from config import options
import torch.nn.functional as F
import torch.nn as nn


def capsnet_loss(img, target, img_reconst, v_length):
    return margin_loss(v_length, target) + 0.0005 * reconstruction_loss(img, img_reconst)


def margin_loss(v_c, labels):
    batch_size = labels.size(0)

    present_error = (F.relu(options.m_plus - v_c)**2).view(batch_size, -1)      # max(0, m_plus-||v_c||)^2
    absent_error = (F.relu(v_c - options.m_minus)**2).view(batch_size, -1)      # max(0, ||v_c||-m_minus)^2

    l_c = labels.float() * present_error + \
          options.lambda_val * (1.0 - labels.float()) * absent_error
    mrgn_loss = l_c.sum(dim=1).mean()
    return mrgn_loss


def reconstruction_loss(x, x_reconst):
    mse_loss = nn.MSELoss()
    loss = mse_loss(x_reconst.view(x_reconst.size(0), -1), x.view(x.size(0), -1))
    return loss