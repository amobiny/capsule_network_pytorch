import os
import warnings
from capsnet import CapsNet
from torch.optim import Adam
import numpy as np
from config import options
import torch
import torch.nn.functional as F
from dataset.mnist import MNIST
from utils.eval_utils import compute_accuracy
from utils.loss_utils import capsnet_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
warnings.filterwarnings("ignore")


def train():
    for epoch in range(options.epochs):
        capsule_net.train()
        train_loss = 0
        global_step = 0
        targets, predictions = [], []

        for batch_id, (data, target) in enumerate(mnist.train_loader):
            data, target = data.cuda(), target.cuda()
            global_step += 1
            target = F.one_hot(target, options.num_classes)

            optimizer.zero_grad()
            y_pred, x_reconst, v_length = capsule_net(data, target)
            loss = capsnet_loss(data, target, x_reconst, v_length)
            loss.backward()
            optimizer.step()

            targets += [target]
            predictions += [y_pred]
            train_loss += loss.item()
            if (batch_id+1) % options.disp_freq == 0:
                train_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))
                print("step: {0}, train_loss: {1:.4f} train_accuracy: {2:.01%}"
                      .format(batch_id+1, train_loss / (options.batch_size*options.disp_freq), train_acc))

            if (batch_id+1) % options.val_freq == 0:
                evaluate()


def evaluate():
    capsule_net.eval()
    test_loss = 0
    for batch_id, (data, target) in enumerate(mnist.test_loader):
        data, target = data.cuda(), target.cuda()
        target = F.one_hot(target, options.num_classes)

        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.data[0]

        if batch_id % 100 == 0:
            print("test accuracy: {}".format(sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                                                 np.argmax(target.data.cpu().numpy(), 1)) / float(
                options.batch_size)))

    print(test_loss / len(mnist.test_loader))


if __name__ == '__main__':
    capsule_net = CapsNet().cuda()
    optimizer = Adam(capsule_net.parameters())
    mnist = MNIST(options.batch_size)
    train()
