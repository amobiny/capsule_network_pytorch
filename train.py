import os
import warnings

from torch.utils.data import DataLoader

from capsnet import CapsuleNet, CapsuleLoss
from torch.optim import Adam
import numpy as np
from config import options
import torch
import torch.nn.functional as F
from dataset.mnist import MNIST
from utils.eval_utils import compute_accuracy

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
warnings.filterwarnings("ignore")


def train():
    for epoch in range(options.epochs):
        capsule_net.train()
        train_loss = 0
        global_step = 0
        targets, predictions = [], []

        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            global_step += 1
            target = F.one_hot(target, options.num_classes)

            optimizer.zero_grad()
            y_pred, x_reconst, v_length = capsule_net(data, target)
            loss = capsule_loss(data, target, v_length, x_reconst)
            loss.backward()
            optimizer.step()

            targets += [target]
            predictions += [y_pred]
            train_loss += loss.item()

            if (batch_id + 1) % options.disp_freq == 0:
                train_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))
                print("step: {0}, train_loss: {1:.4f} train_accuracy: {2:.01%}"
                      .format(batch_id + 1, train_loss / (options.batch_size * options.disp_freq), train_acc))
                train_loss = 0

            if (batch_id + 1) % options.val_freq == 0:
                evaluate()


@torch.no_grad()
def evaluate():
    capsule_net.eval()
    test_loss = 0
    targets, predictions = [], []

    for batch_id, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        target = F.one_hot(target, options.num_classes)
        y_pred, x_reconst, v_length = capsule_net(data, target)
        loss = capsule_loss(data, target, v_length, x_reconst)

        targets += [target]
        predictions += [y_pred]
        test_loss += loss

    test_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))
    print("validation_loss: {0:.4f}, validation_accuracy: {1:.01%}"
          .format(test_loss / len(test_loader), test_acc))


if __name__ == '__main__':
    capsule_net = CapsuleNet(options)
    capsule_net.cuda()
    print("# parameters:", sum(param.numel() for param in capsule_net.parameters()))
    capsule_loss = CapsuleLoss()
    optimizer = Adam(capsule_net.parameters())

    train_dataset = MNIST(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = MNIST(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    train()
