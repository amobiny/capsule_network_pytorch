import torch
import torch.nn as nn
import torch.nn.functional as F
from config import options
from utils.other_utils import squash, coord_addition


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias))

    def forward(self, x):
        return self.net(x)


class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, cap_dim, num_cap_map, add_coord):
        super(PrimaryCapsLayer, self).__init__()

        self.capsule_dim = cap_dim
        self.num_cap_map = num_cap_map
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.add_coord = add_coord

    def forward(self, x):
        batch_size = x.size(0)
        outputs = self.conv_out(x)
        map_dim = outputs.size(-1)
        outputs = outputs.view(batch_size, self.capsule_dim, self.num_cap_map, map_dim, map_dim)
        # [bs, 8 (or 10), 32, 6, 6]
        if self.add_coord:
            outputs = coord_addition(outputs)  # [bs, 10, 32, 6, 6]
            outputs = outputs.view(batch_size, self.capsule_dim + 2, self.num_cap_map, -1).transpose(1, 2).transpose(2,
                                                                                                                     3)
            # [bs, 32, 36, 10]
        else:
            outputs = outputs.view(batch_size, self.capsule_dim, self.num_cap_map, -1).transpose(1, 2).transpose(2, 3)
            # [bs, 32, 36, 8]
        outputs = squash(outputs)
        return outputs


class DigitCapsLayer(nn.Module):
    def __init__(self, num_digit_cap, num_prim_cap, num_prim_map, in_cap_dim, out_cap_dim, num_iterations):
        super(DigitCapsLayer, self).__init__()
        self.num_prim_cap = num_prim_cap
        self.num_iterations = num_iterations
        self.out_cap_dim = out_cap_dim
        if options.share_weight:
            self.W = nn.Parameter(torch.randn(1, num_prim_map, 1, num_digit_cap, out_cap_dim, in_cap_dim))
            # [1, 32, 1, 10, 16, 8]
        else:
            self.W = nn.Parameter(torch.randn(1, num_prim_map, num_prim_cap, num_digit_cap, out_cap_dim, in_cap_dim))
            # [1, 32, 36, 10, 16, 8]

    def forward(self, x):
        batch_size = x.size(0)  # [bs, num_prim_map, num_prim_cap, primary_cap_dim]
        if options.share_weight:
            W = self.W.repeat(batch_size, 1, self.num_prim_cap, 1, 1, 1)
        else:
            W = self.W.repeat(batch_size, 1, 1, 1, 1, 1)

        u = x[:, :, :, None, :, None].repeat(1, 1, 1, options.num_classes, 1, 1)
        u_hat = torch.matmul(W, u)
        u_hat = u_hat.view(batch_size, -1, options.num_classes, self.out_cap_dim, 1)
        # [10, 32, 36, 10, 16, 1] --> [10, 1152, 10, 16, 1]

        b_ij = torch.zeros(batch_size, u_hat.size(1), u_hat.size(2), 1, 1).cuda()
        for i in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=1)  # it must be 2, but works with 1!
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            outputs = squash(s_j, dim=-2)

            if i != self.num_iterations - 1:
                outputs_tiled = outputs.repeat(1, u_hat.size(1), 1, 1, 1)
                u_produce_v = torch.matmul(u_hat.transpose(-1, -2), outputs_tiled)
                b_ij = b_ij + u_produce_v
        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, args):
        super(CapsuleNet, self).__init__()
        self.args = args

        # convolution layer
        self.conv1 = nn.Conv2d(in_channels=args.img_c, out_channels=args.f1, kernel_size=args.k1, stride=1)

        # primary capsule layer
        assert args.f2 % args.primary_cap_dim == 0
        self.num_prim_map = int(args.f2 / args.primary_cap_dim)
        self.primary_capsules = PrimaryCapsLayer(in_channels=args.f1, out_channels=args.f2,
                                                 kernel_size=args.k2, stride=1,
                                                 cap_dim=args.primary_cap_dim,
                                                 num_cap_map=self.num_prim_map,
                                                 add_coord=args.add_coord)
        self.digit_capsules = DigitCapsLayer(num_digit_cap=args.num_classes,
                                             num_prim_cap=12 * 12,
                                             num_prim_map=self.num_prim_map,
                                             in_cap_dim=args.primary_cap_dim if not args.add_coord
                                             else args.primary_cap_dim + 2,
                                             out_cap_dim=args.digit_cap_dim,
                                             num_iterations=args.num_iterations)

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
        x = self.digit_capsules(x).squeeze(1).squeeze(-1)

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
        present_error = F.relu(self.args.m_plus - v_c, inplace=True) ** 2  # max(0, m_plus-||v_c||)^2
        absent_error = F.relu(v_c - self.args.m_minus, inplace=True) ** 2  # max(0, ||v_c||-m_minus)^2

        l_c = labels.float() * present_error + self.args.lambda_val * (1. - labels.float()) * absent_error
        margin_loss = l_c.sum()

        reconstruction_loss = 0
        if self.args.add_decoder:
            assert torch.numel(images) == torch.numel(reconstructions)
            images = images.view(reconstructions.size()[0], -1)
            reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + self.args.alpha * reconstruction_loss) / images.size(0)
