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
            self.W = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
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
            u_hat = x[None, :, :, None, :] @ self.W[:, None, :, :, :]

            b_ij = torch.zeros(*u_hat.size()).cuda()
            for i in range(self.num_iterations):
                c_ij = F.softmax(b_ij, dim=2)
                outputs = self.squash((c_ij * u_hat).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    u_produce_v = (u_hat * outputs).sum(dim=-1, keepdim=True)
                    b_ij = b_ij + u_produce_v
        else:        # for primary capsules
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, args):
        super(CapsuleNet, self).__init__()
        self.num_cls = args.num_classes
        self.conv1 = nn.Conv2d(in_channels=args.img_c, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=args.num_classes, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * self.num_cls, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        # classes = F.softmax(classes, dim=-1)

        _, max_length_indices = classes.max(dim=1)
        y_pred = torch.eye(self.num_cls).cuda().index_select(dim=0, index=max_length_indices)

        if y is None:
            y = y_pred

        reconstructions = self.decoder((x * y[:, :, None].float()).view(x.size(0), -1))

        return y_pred, reconstructions, classes


class CapsuleLoss(nn.Module):
    def __init__(self, args):
        super(CapsuleLoss, self).__init__()
        self.args = args
        self.reconstruction_loss = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(self.args.m_plus - classes, inplace=True) ** 2
        right = F.relu(classes - self.args.m_minus, inplace=True) ** 2

        margin_loss = labels.float() * left + self.args.lambda_val * (1. - labels.float()) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
