import torch
import torch.nn as nn
import torch.nn.functional as F
from config import options


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=options.num_iterations):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:   # for digit capsules
            self.W = nn.Parameter(torch.randn(1, num_route_nodes, num_capsules, out_channels, in_channels))
        else:                       # for primary capsules
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, input_tensor, dim=-1, epsilon=1e-7):
        squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
        safe_norm = torch.sqrt(squared_norm + epsilon)
        scale = squared_norm / (1 + squared_norm)
        unit_vector = input_tensor / safe_norm
        return scale * unit_vector

    def forward(self, x):
        if self.num_route_nodes != -1:      # for digit capsules
            batch_size = x.size(0)
            W = self.W.repeat(batch_size, 1, 1, 1, 1)
            u = x[:, :, None, :, None].repeat(1, 1, options.num_classes, 1, 1)
            u_hat = torch.matmul(W, u)

            b_ij = torch.zeros(batch_size, u_hat.size(1), u_hat.size(2), 1, 1).cuda()
            for i in range(self.num_iterations):
                c_ij = F.softmax(b_ij, dim=1)   # it must be 2, but works with 1!
                s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
                outputs = self.squash(s_j, dim=-2)

                if i != self.num_iterations - 1:
                    outputs_tiled = outputs.repeat(1, u_hat.size(1), 1, 1, 1)
                    u_produce_v = torch.matmul(u_hat.transpose(-1, -2), outputs_tiled)
                    b_ij = b_ij + u_produce_v
        else:        # for primary capsules
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, args):
        super(CapsuleNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(in_channels=args.img_c, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=args.num_classes, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        if args.add_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(16 * args.num_classes, args.h1),
                nn.ReLU(inplace=True),
                nn.Linear(args.h1, args.h2),
                nn.ReLU(inplace=True),
                nn.Linear(args.h2, args.img_h * args.img_w),
                nn.Sigmoid()
            )

    def forward(self, imgs, y=None):
        x = F.relu(self.conv1(imgs), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze()

        v_length = (x ** 2).sum(dim=-1) ** 0.5

        _, y_pred = v_length.max(dim=1)
        y_pred_ohe = F.one_hot(y_pred, self.args.num_classes)

        if y is None:
            y = y_pred_ohe

        img_reconst = torch.zeros_like(imgs)
        if self.args.add_decoder:
            img_reconst = self.decoder((x * y[:, :, None].float()).view(x.size(0), -1))

        return y_pred_ohe, img_reconst, v_length


class CapsuleLoss(nn.Module):
    def __init__(self, args):
        super(CapsuleLoss, self).__init__()
        self.args = args
        self.reconstruction_loss = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, v_c, reconstructions):
        present_error = F.relu(self.args.m_plus - v_c, inplace=True) ** 2   # max(0, m_plus-||v_c||)^2
        absent_error = F.relu(v_c - self.args.m_minus, inplace=True) ** 2   # max(0, ||v_c||-m_minus)^2

        l_c = labels.float() * present_error + self.args.lambda_val * (1. - labels.float()) * absent_error
        margin_loss = l_c.sum()

        reconstruction_loss = 0
        if self.args.add_decoder:
            assert torch.numel(images) == torch.numel(reconstructions)
            images = images.view(reconstructions.size()[0], -1)
            reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + self.args.alpha * reconstruction_loss) / images.size(0)
