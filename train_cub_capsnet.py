import os
import warnings
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore")
from datetime import datetime
from torch.utils.data import DataLoader

from cub_capsnet import CapsuleNet, CapsuleLoss

from torch.optim import Adam
import numpy as np
from config import options
import torch
import torch.nn.functional as F
from utils.eval_utils import compute_accuracy
from utils.logger_utils import Logger
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
theta_c = 0.5


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    global_step = 0
    best_loss = 100
    best_acc = 0

    for epoch in range(options.epochs):
        log_string('**' * 30)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        capsule_net.train()

        train_loss = np.zeros(2)
        targets, predictions, predictions_crop = [], [], []

        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            global_step += 1
            target_ohe = F.one_hot(target, options.num_classes)

            optimizer.zero_grad()
            y_pred, x_reconst, v_length, attention_map = capsule_net(data, target_ohe)
            loss = capsule_loss(data, target_ohe, v_length, x_reconst)
            loss.backward()
            optimizer.step()

            targets += [target_ohe]
            predictions += [y_pred]
            train_loss[0] += loss.item()

            ##################################
            # Attention Cropping
            ##################################
            empty_map_count = 0
            one_nonzero_count = 0
            width_count = 0
            height_count = 0
            with torch.no_grad():
                crop_mask = F.interpolate(attention_map, size=(data.size(2), data.size(3)), mode='bilinear',
                                          align_corners=True) > theta_c
                crop_images = []
                for batch_index in range(crop_mask.size(0)):
                    if torch.sum(crop_mask[batch_index]) == 0:
                        height_min, width_min = 0, 0
                        height_max, width_max = 200, 200
                        # print('0, batch: {}, map: {}'.format(batch_index, map_index))
                        empty_map_count += 1
                    else:
                        nonzero_indices = torch.nonzero(crop_mask[batch_index])
                        if nonzero_indices.size(0) == 1:
                            height_min, width_min = 0, 0
                            height_max, width_max = 200, 200
                            # print('1, batch: {}, map: {}'.format(batch_index, map_index))
                            one_nonzero_count += 1
                        else:
                            height_min = nonzero_indices[:, 0].min()
                            height_max = nonzero_indices[:, 0].max()
                            width_min = nonzero_indices[:, 1].min()
                            width_max = nonzero_indices[:, 1].max()
                        if width_min == width_max:
                            if width_min == 0:
                                width_max += 1
                            else:
                                width_min -= 1
                            width_count += 1
                        if height_min == height_max:
                            if height_min == 0:
                                height_max += 1
                            else:
                                height_min -= 1
                            height_count += 1
                    crop_images.append(F.upsample_bilinear(
                        data[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                        size=options.img_h))
                # print('Batch {} :  empty map: {},  one nonzero idx: {}, width_issue: {}, height_issue: {}'
                #       .format(i, empty_map_count, one_nonzero_count, width_count, height_count))

            crop_images = torch.cat(crop_images, dim=0)

            # crop images forward
            y_pred, _, v_length, _ = capsule_net(crop_images.cuda(), target_ohe)
            loss = capsule_loss(data, target_ohe, v_length, x_reconst)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions_crop += [y_pred.float()]
            train_loss[1] += loss.item()

            if (batch_id + 1) % options.disp_freq == 0:
                train_loss /= (options.batch_size * options.disp_freq)
                train_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))
                train_acc_crop = compute_accuracy(torch.cat(targets), torch.cat(predictions_crop))
                log_string("epoch: {0}, step: {1}, (Raw): loss: {2:.4f} acc: {3:.02%}, "
                           "(Crop): loss: {2:.4f} acc: {3:.02%}"
                           .format(epoch+1, batch_id+1, train_loss[0], train_acc, train_loss[1], train_acc_crop))
                info = {'loss/raw': train_loss[0],
                        'loss/crop': train_loss[1],
                        'accuracy/raw': train_acc,
                        'accuracy/crop': train_acc_crop}
                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, global_step)
                train_loss = np.zeros(2)
                targets, predictions, predictions_crop = [], [], []

            if (batch_id + 1) % options.val_freq == 0:
                log_string('--' * 30)
                log_string('Evaluating at step #{}'.format(global_step))
                best_loss, best_acc = evaluate(best_loss=best_loss,
                                               best_acc=best_acc,
                                               global_step=global_step)
                capsule_net.train()


@torch.no_grad()
def evaluate(**kwargs):
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    global_step = kwargs['global_step']

    capsule_net.eval()
    test_loss = np.zeros(3)
    targets, predictions_raw, predictions_crop, predictions_combined = [], [], [], []

    empty_map_count = 0
    one_nonzero_count = 0
    width_count = 0
    height_count = 0

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            target_ohe = F.one_hot(target, options.num_classes)
            y_pred_raw, x_reconst, v_length, attention_map = capsule_net(data, target_ohe)
            loss = capsule_loss(data, target_ohe, v_length, x_reconst)
            targets += [target_ohe]
            predictions_raw += [y_pred_raw]
            test_loss[0] += loss

            ##################################
            # Object Localization and Refinement
            ##################################
            crop_mask = F.upsample_bilinear(attention_map, size=(data.size(2), data.size(3))) > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                if torch.sum(crop_mask[batch_index]) == 0:
                    height_min, width_min = 0, 0
                    height_max, width_max = 200, 200
                    # print('0, batch: {}, map: {}'.format(batch_index, map_index))
                    empty_map_count += 1
                else:
                    nonzero_indices = torch.nonzero(crop_mask[batch_index])
                    if nonzero_indices.size(0) == 1:
                        height_min, width_min = 0, 0
                        height_max, width_max = 200, 200
                        # print('1, batch: {}, map: {}'.format(batch_index, map_index))
                        one_nonzero_count += 1
                    else:
                        height_min = nonzero_indices[:, 0].min()
                        height_max = nonzero_indices[:, 0].max()
                        width_min = nonzero_indices[:, 1].min()
                        width_max = nonzero_indices[:, 1].max()
                    if width_min == width_max:
                        if width_min == 0:
                            width_max += 1
                        else:
                            width_min -= 1
                        width_count += 1
                    if height_min == height_max:
                        if height_min == 0:
                            height_max += 1
                        else:
                            height_min -= 1
                        height_count += 1
                crop_images.append(F.upsample_bilinear(
                    data[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=options.img_h))
            crop_images = torch.cat(crop_images, dim=0)

            y_pred_crop, _, v_length, c_maps = capsule_net(crop_images, target_ohe)
            loss = capsule_loss(data, target_ohe, v_length, x_reconst)
            predictions_crop += [y_pred_crop]
            test_loss[1] += loss

            # final prediction
            y_pred_combined = (y_pred_raw + y_pred_crop) / 2
            predictions_combined += [y_pred_combined]

        test_loss /= (len(test_loader) * options.batch_size)
        test_acc_raw = compute_accuracy(torch.cat(targets), torch.cat(predictions_raw))
        test_acc_crop = compute_accuracy(torch.cat(targets), torch.cat(predictions_crop))
        test_acc_combined = compute_accuracy(torch.cat(targets), torch.cat(predictions_combined))

        # check for improvement
        loss_str, acc_str = '', ''
        if test_loss[0] <= best_loss:
            loss_str, best_loss = '(improved)', test_loss[0]
        if test_acc_combined >= best_acc:
            acc_str, best_acc = '(improved)', test_acc_combined

        # display
        log_string(" - (Raw)      loss: {0:.4f}, accuracy: {1:.02%}"
                   .format(test_loss[0], test_acc_raw))
        log_string(" - (Crop)     loss: {0:.4f}, accuracy: {1:.02%}"
                   .format(test_loss[1], test_acc_crop))
        log_string(" - (Combined) loss: {0:.4f} {1}, accuracy: {2:.02%}{3}"
                   .format(test_loss[2], loss_str, test_acc_combined, acc_str))
        # write to TensorBoard
        info = {'loss/raw': test_loss[0],
                'loss/crop': test_loss[0],
                'accuracy/raw': test_acc_raw,
                'accuracy/crop': test_acc_crop,
                'accuracy/combined': test_acc_combined}
        for tag, value in info.items():
            test_logger.scalar_summary(tag, value, global_step)

        # save checkpoint model
        state_dict = capsule_net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        save_path = os.path.join(model_dir, '{}.ckpt'.format(global_step))
        torch.save({
            'global_step': global_step,
            'acc': test_acc_combined,
            'save_dir': model_dir,
            'state_dict': state_dict},
            save_path)
        log_string('Model saved at: {}'.format(save_path))
        log_string('--' * 30)
        return best_loss, best_acc


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # bkp of model def
    os.system('cp {}/hough_capsnet_multichannel.py {}'.format(BASE_DIR, save_dir))
    # bkp of train procedure
    os.system('cp {}/train.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/config.py {}'.format(BASE_DIR, save_dir))

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
    # capsule_net = nn.DataParallel(capsule_net)

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
        os.system('cp {}/dataset/mnist.py {}'.format(BASE_DIR, save_dir))
    elif options.data_name == 'fashion_mnist':
        from dataset.fashion_mnist import FashionMNIST as data
        os.system('cp {}/dataset/fashion_mnist.py {}'.format(BASE_DIR, save_dir))
    elif options.data_name == 't_mnist':
        from dataset.mnist_translate import MNIST as data
        os.system('cp {}/dataset/mnist_translate.py {}'.format(BASE_DIR, save_dir))
    elif options.data_name == 'c_mnist':
        from dataset.mnist_clutter import MNIST as data
        os.system('cp {}/dataset/mnist_clutter.py {}'.format(BASE_DIR, save_dir))
    elif options.data_name == 'cub':
        from dataset.dataset_CUB import CUB as data
        os.system('cp {}/dataset/dataset_CUB.py {}'.format(BASE_DIR, save_dir))

    train_dataset = data(mode='train', data_len=100)
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data(mode='test', data_len=100)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_dataset), len(test_dataset)))
    train_logger = Logger(os.path.join(logs_dir, 'train'))
    test_logger = Logger(os.path.join(logs_dir, 'test'))

    train()
