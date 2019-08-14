import argparse
import cv2
import os

import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.lip import LipTrainDataset, LipValDataset
from datasets.coco2017 import COCO2017TrainDataset, COCO2017ValDataset
from datasets.transformations import SinglePersonRotate, SinglePersonCropPad, SinglePersonFlip, SinglePersonBodyMasking,\
    ChannelPermutation
from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.single_person_pose_with_mobilenet import SinglePersonPoseEstimationWithMobileNet
from modules.loss import l2_loss
from modules.load_state import load_state, load_from_mobilenet
import val_lip, val_coco

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader

DATASET = 'COCO2017'
dtst_train = LipTrainDataset if DATASET == 'LIP' else COCO2017TrainDataset
dtst_val = LipValDataset if DATASET == 'LIP' else COCO2017ValDataset
evlt = val_lip.evaluate if DATASET == 'lIP' else val_coco.evaluate

STRIDE = 8
SIGMA = 7


def validate2(epoch, net, val_dataset, scheduler):
    print('Validation...')
    net.eval()
    predictions_name = '{}/val_{}_results.csv'.format(checkpoints_folder, DATASET)
    pck = evlt(val_dataset, predictions_name, net)
    val_loss = 100 - pck[-1][-1]  # 100 - avg_pckh
    print('Val loss: {}'.format(val_loss))
    scheduler.step(val_loss, epoch)
    net.train()


def validate2(epoch, net, loader, scheduler, N_losses):
    print('Validation...')
    net.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].cuda()
            keypoint_maps = batch['keypoint_maps'].cuda()

            stages_output = net(images)

            loss = 0.0
            for loss_idx in range(N_losses):
                loss += l2_loss(stages_output[loss_idx], keypoint_maps, len(images))

    print('Val loss: {}'.format(loss))
    net.train()


def train(images_folder, num_refinement_stages, base_lr, batch_size, batches_per_iter,
          num_workers, checkpoint_path, weights_only, from_mobilenet, checkpoints_folder,
          log_after, checkpoint_after):
    net = SinglePersonPoseEstimationWithMobileNet(num_refinement_stages, num_heatmaps=18).cuda()

    train_dataset = dtst_train(images_folder, STRIDE, SIGMA,
                              transform=transforms.Compose([
                                   SinglePersonBodyMasking(),
                                   ChannelPermutation(),
                                   SinglePersonRotate(pad=(128, 128, 128), max_rotate_degree=40),
                                   SinglePersonCropPad(pad=(128, 128, 128), crop_x=256, crop_y=256),
                                   SinglePersonFlip()]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = dtst_val(images_folder, STRIDE, SIGMA)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    optimizer = optim.Adam([
        {'params': get_parameters_conv(net.model, 'weight')},
        {'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_bn(net.initial_stage, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
        {'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
    ], lr=base_lr, weight_decay=5e-4)

    num_iter = 0
    current_epoch = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=1e-2, verbose=True)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        if from_mobilenet:
            load_from_mobilenet(net, checkpoint)
        else:
            load_state(net, checkpoint)
            if not weights_only:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                num_iter = checkpoint['iter']
                num_iter = num_iter // log_after * log_after  # round iterations, to print proper loss when resuming
                current_epoch = checkpoint['current_epoch']+1

    net = DataParallel(net)
    net.train()
    for epochId in range(current_epoch, 100):
        print('Epoch: {}'.format(epochId))
        N_losses = num_refinement_stages + 1
        total_losses = [0] * N_losses  # heatmaps loss per stage
        for batch in train_loader:
            images = batch['image'].cuda()
            keypoint_maps = batch['keypoint_maps'].cuda()

            stages_output = net(images)

            losses = []
            for loss_idx in range(N_losses):
                loss = l2_loss(stages_output[loss_idx], keypoint_maps, len(images))
                losses.append(loss)
                total_losses[loss_idx] += loss.item()

            optimizer.zero_grad()
            loss = losses[1]
            for i in range(N_losses):
                loss += losses[i]
            loss.backward()
            optimizer.step()

            num_iter += 1

            if num_iter % log_after == 0:
                print('Iter: {}'.format(num_iter))
                # for loss_idx in range(N_losses):
                    # print('\n'.join(['stage{}_heatmaps_loss: {}']).format(
                    #       loss_idx + 1, total_losses[loss_idx] / log_after))
                for loss_idx in range(N_losses):
                    total_losses[loss_idx] = 0
                validate2(epochId, net, val_loader, scheduler, N_losses)

        snapshot_name = '{}/{}_epoch_last.pth'.format(checkpoints_folder, DATASET)
        torch.save({'state_dict': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iter': num_iter,
                    'current_epoch': epochId},
                     snapshot_name)

        if epochId % checkpoint_after == 0:
            snapshot_name = '{}/{}_epoch_{}.pth'.format(checkpoints_folder, DATASET, epochId)
            torch.save({'state_dict': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iter': num_iter,
                        'current_epoch': epochId},
                        snapshot_name)

        validate2(epochID, net, val_loader, scheduler, N_losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', type=str, required=True, help='path to dataset folder')
    parser.add_argument('--num-refinement-stages', type=int, default=5, help='number of refinement stages')
    parser.add_argument('--base-lr', type=float, default=4e-5, help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--batches-per-iter', type=int, default=1, help='number of batches to accumulate gradient from')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint to continue training from')
    parser.add_argument('--from-mobilenet', action='store_true',
                        help='load weights from mobilenet feature extractor')
    parser.add_argument('--weights-only', action='store_true',
                        help='just initialize layers with pre-trained weights and start training from the beginning')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='experiment name to create folder for checkpoints')
    parser.add_argument('--log-after', type=int, default=100, help='number of iterations to print train loss')
    parser.add_argument('--checkpoint-after', type=int, default=10,
                        help='number of epochs to save checkpoint')
    args = parser.parse_args()

    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    train(args.dataset_folder, args.num_refinement_stages, args.base_lr, args.batch_size,
          args.batches_per_iter, args.num_workers, args.checkpoint_path, args.weights_only, args.from_mobilenet,
          checkpoints_folder, args.log_after, args.checkpoint_after)
