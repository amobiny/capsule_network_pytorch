import torch
import torch.nn as nn
import torch.nn.functional as F
from config import options


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1)

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=2, padding=0)
                                       for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=options.num_classes, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        with torch.no_grad():
            b_ij = torch.zeros(options.batch_size, self.num_routes, self.num_capsules, 1, 1).cuda()
            for iteration in range(options.num_iterations):
                c_ij = F.softmax(b_ij, dim=2)

                s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
                v_j = self.squash(s_j)

                if iteration < options.num_iterations - 1:
                    a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                    # b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)
                    b_ij += a_ij

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.reconstraction_layers = nn.Sequential(nn.Linear(16 * options.num_classes, 512),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(512, 1024),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(1024, 784),
                                                   nn.Sigmoid()
                                                   )

    def forward(self, output, y, is_train=True):
        epsilon = 1e-9
        v_length = torch.sqrt((output ** 2).sum(2) + epsilon)
        # classes = F.softmax(classes)

        _, y_pred = v_length.max(dim=1)
        y_pred_ohe = F.one_hot(y_pred.squeeze(), options.num_classes)
        if is_train:
            target_to_reconstruct = y
        else:
            target_to_reconstruct = y_pred_ohe

        reconstructions = self.reconstraction_layers(
            (output * target_to_reconstruct[:, :, None, None].float()).view(output.size(0), -1))
        image_reconstructed = reconstructions.view(-1, 1, 28, 28)

        return image_reconstructed, y_pred_ohe, v_length


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()

    def forward(self, data, target=None):
        output = self.conv_layer(data)
        output = self.primary_capsules(output)
        output = self.digit_capsules(output)
        x_reconst, y_pred_ohe, v_length = self.decoder(output, target, is_train=self.training)
        return y_pred_ohe, x_reconst, v_length
