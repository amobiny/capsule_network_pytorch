import torch
import torch.nn as nn
import torch.nn.functional as F
from config import options
from utils.other_utils import squash, coord_addition
from models import *
import numpy as np


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
            outputs = coord_addition(outputs, shuffle=options.shuffle_coords)   # [bs, 10, 32, 6, 6]
            outputs = outputs.view(batch_size, self.capsule_dim+2, self.num_cap_map, -1).transpose(1, 2).transpose(2, 3)
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
        map_size = int(np.sqrt(u.size(2)))
        c_maps = c_ij.reshape(batch_size, u.size(1), map_size, map_size, options.num_classes) # (batch_size, 32, 20, 20, 200)
        return outputs.squeeze(1).squeeze(-1), c_maps


class CapsuleNet(nn.Module):
    def __init__(self, args):
        super(CapsuleNet, self).__init__()
        self.args = args

        # convolution layer
        if options.feature_extractor == 'resnet':
            net = resnet50(pretrained=True)
            self.num_features = 1024
        elif options.feature_extractor == 'densent':
            net = densenet121(pretrained=True)
            self.num_features = 1024
        elif options.feature_extractor == 'inception':
            net = inception_v3(pretrained=True)
            self.num_features = 768

        self.features = net.get_features()

        self.conv1 = Conv2dSame(self.num_features, args.f1, 1)

        # primary capsule layer
        assert args.f2 % args.primary_cap_dim == 0
        self.num_prim_map = int(args.f2 / args.primary_cap_dim)
        self.primary_capsules = PrimaryCapsLayer(in_channels=args.f1, out_channels=args.f2,
                                                 kernel_size=args.k2, stride=1,
                                                 cap_dim=args.primary_cap_dim,
                                                 num_cap_map=self.num_prim_map,
                                                 add_coord=args.add_coord)
        self.digit_capsules = DigitCapsLayer(num_digit_cap=args.num_classes,
                                             num_prim_cap=18 * 18,
                                             num_prim_map=self.num_prim_map,
                                             in_cap_dim=args.primary_cap_dim if not args.add_coord
                                             else args.primary_cap_dim+2,
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
        x = self.features.forward(imgs)
        x = F.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x, attention_maps = self.digit_capsules(x)

        v_length = (x ** 2).sum(dim=-1) ** 0.5

        _, y_pred = v_length.max(dim=1)
        y_pred_ohe = F.one_hot(y_pred, self.args.num_classes)

        if y is None:
            y = y_pred_ohe

        img_reconst = torch.zeros_like(imgs)
        if self.args.add_decoder:
            img_reconst = self.decoder((x * y[:, :, None].float()).view(x.size(0), -1))

        # Generate Attention Map
        batch_size, NUM_MAPS, H, W, _ = attention_maps.size()
        if self.training:
            # Randomly choose one of attention maps ck
            k_indices = np.random.randint(NUM_MAPS, size=batch_size)
            attention_map = attention_maps[torch.arange(batch_size), k_indices, :, :, y.argmax(dim=1)].to(torch.device("cuda"))
            if len(attention_map.size()) == 3:  # for batch_size=1
                attention_map = attention_map.unsqueeze(0)
            # (B, 1, H, W)
        else:
            attention_maps = attention_maps[torch.arange(batch_size), :, :, :, y.argmax(dim=1)].to(torch.device("cuda"))
            # (B, NUM_MAPS, H, W)

            # Object Localization Am = mean(sum(Ak))
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        # Normalize Attention Map
        attention_map = attention_map.view(batch_size, -1)  # (B, H * W)
        attention_map_max, _ = attention_map.max(dim=1, keepdim=True)  # (B, 1)
        attention_map_min, _ = attention_map.min(dim=1, keepdim=True)  # (B, 1)
        attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min)  # (B, H * W)
        attention_map = attention_map.view(batch_size, 1, H, W)  # (B, 1, H, W)

        return y_pred_ohe, img_reconst, v_length, attention_map


class CapsuleLoss(nn.Module):
    def __init__(self, args):
        super(CapsuleLoss, self).__init__()
        self.args = args

    def forward(self, images, labels, v_c, reconstructions):
        present_error = F.relu(self.args.m_plus - v_c, inplace=True) ** 2  # max(0, m_plus-||v_c||)^2
        absent_error = F.relu(v_c - self.args.m_minus, inplace=True) ** 2  # max(0, ||v_c||-m_minus)^2

        l_c = labels.float() * present_error + self.args.lambda_val * (1. - labels.float()) * absent_error
        margin_loss = l_c.sum(dim=1).mean()

        reconstruction_loss = 0
        if self.args.add_decoder:
            assert torch.numel(images) == torch.numel(reconstructions)
            images = images.view(reconstructions.size()[0], -1)
            reconstruction_loss = torch.mean((reconstructions - images) ** 2)

        return margin_loss + self.args.alpha * reconstruction_loss
