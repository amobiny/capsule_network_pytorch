import os
import warnings
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader

from hough_capsnet_multichannel import CapsuleNet, CapsuleLoss

from torch.optim import Adam
import numpy as np
from config import options
import torch
import torch.nn.functional as F
from utils.eval_utils import compute_accuracy
from utils.logger_utils import Logger
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


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

    test_loss /= (len(test_loader) * options.batch_size)
    test_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))

    # display
    log_string("test_loss: {0:.7f}, test_accuracy: {1:.04%}"
               .format(test_loss, test_acc))


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))

    LOG_FOUT = open(os.path.join(save_dir, 'log_inference.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    # bkp of inference
    os.system('cp {}/inference.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    capsule_net = CapsuleNet(options)
    log_string('Model Generated.')
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in capsule_net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    capsule_net.cuda()
    capsule_net = nn.DataParallel(capsule_net)

    ##################################
    # Load the trained model
    ##################################
    ckpt = options.load_model_path
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    capsule_net.load_state_dict(state_dict)
    log_string('Model successfully loaded from {}'.format(ckpt))

    ##################################
    # Loss and Optimizer
    ##################################

    capsule_loss = CapsuleLoss(options)
    optimizer = Adam(capsule_net.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################
    if options.data_name == 'mnist':
        from dataset.mnist import MNIST as data
    elif options.data_name == 'fashion_mnist':
        from dataset.fashion_mnist import FashionMNIST as data
    elif options.data_name == 't_mnist':
        from dataset.mnist_translate import MNIST as data

    train_dataset = data(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')

    evaluate()
