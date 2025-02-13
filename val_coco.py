import argparse
import cv2
import numpy as np
import sys

import torch

from datasets.lip import LipValDataset
from models.single_person_pose_with_mobilenet import SinglePersonPoseEstimationWithMobileNet
from modules.calc_pckh import calc_pckh
from modules.load_state import load_state
from modules.calc_pckh import calc_pckh_2

N_KPTS = 17


def extract_keypoints(heatmap, min_confidence=-100):
    ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    if heatmap[ind] < min_confidence:
        ind = (-1, -1)
    else:
        ind = (int(ind[1]), int(ind[0]))
    return ind


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def infer(net, img, scales, base_height, stride, img_mean=[128, 128, 128], img_scale=1/256):
    height, width, _ = img.shape
    scales_ratios = [scale * base_height / max(height, width) for scale in scales]
    avg_heatmaps = np.zeros((height, width, 17), dtype=np.float32)

    for ratio in scales_ratios:
        resized_img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        max_side = max(resized_img.shape[0], resized_img.shape[1])

        padded_img = np.ones((max_side, max_side, 3), dtype=np.uint8) * img_mean
        x_offset = (padded_img.shape[1] - resized_img.shape[1]) // 2
        y_offset = (padded_img.shape[0] - resized_img.shape[0]) // 2
        padded_img[y_offset:y_offset + resized_img.shape[0], x_offset:x_offset + resized_img.shape[1], :] = resized_img
        padded_img = normalize(padded_img, img_mean, img_scale)
        pad = [y_offset, x_offset,
               padded_img.shape[0] - resized_img.shape[0] - y_offset,
               padded_img.shape[1] - resized_img.shape[1] - x_offset]

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        stages_output = net(tensor_img)

        heatmaps = np.transpose(stages_output[-1].squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

    return avg_heatmaps


def evaluate(dataset, output_name, net, multiscale=False, visualize=False):
    net.eval()

    base_height = 256
    scales = [1]
    if multiscale:
        scales = [0.75, 1.0, 1.25]
    stride = 8

    for sample in dataset:
        file_name = sample['file_name']
        img = sample['image']

        avg_heatmaps = infer(net, img, scales, base_height, stride)

        flip = False
        if flip:
            flipped_img = cv2.flip(img, 1)
            flipped_avg_heatmaps = infer(net, flipped_img, scales, base_height, stride)
            orig_order = [0, 1, 2, 10, 11, 12]
            flip_order = [5, 4, 3, 15, 14, 13]
            for r, l in zip(orig_order, flip_order):
                flipped_avg_heatmaps[:, :, r], flipped_avg_heatmaps[:, :, l] =\
                    flipped_avg_heatmaps[:, :, l].copy(), flipped_avg_heatmaps[:, :, r].copy()
            avg_heatmaps = (avg_heatmaps + flipped_avg_heatmaps[:, ::-1]) / 2

        all_keypoints = []
        for kpt_idx in range(N_KPTS):
            all_keypoints.append(extract_keypoints(avg_heatmaps[:, :, kpt_idx]))

        # res_file.write('{}'.format(file_name))
        # for id in range(N_KPTS):
        #     val = [int(all_keypoints[id][0]), int(all_keypoints[id][1])]
        #     if val[0] == -1:
        #         val[0], val[1] = 'nan', 'nan'
        #     res_file.write(',{},{}'.format(val[0], val[1]))
        # res_file.write('\n')

        if visualize:
            kpt_names = ['r_ank', 'r_kne', 'r_hip', 'l_hip', 'l_kne', 'l_ank', 'pel', 'spi', 'nec', 'hea',
                         'r_wri', 'r_elb', 'r_sho', 'l_sho', 'l_elb', 'l_wri']
            colors = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0),
                      (0, 255, 0), (0, 255, 0), (0, 255, 0),
                      (255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255)]
            for id in range(len(all_keypoints)):
                keypoint = all_keypoints[id]
                if keypoint[0] != -1:
                    radius = 3
                    if colors[id] == (255, 0, 0):
                        cv2.circle(img, (int(keypoint[0]), int(keypoint[1])),
                                   radius + 2, (255, 0, 0), -1)
                    else:
                        cv2.circle(img, (int(keypoint[0]), int(keypoint[1])),
                                   radius, colors[id], -1)
            cv2.imshow('keypoints', img)
            key = cv2.waitKey()
            if key == 27:  # esc
                sys.exit()

    pck = calc_pckh_2(val_dataset.labels_file_path, predictions_name, eval_num=1000)

    return pck


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', type=str, required=True, help='path to dataset folder')
    parser.add_argument('--output-name', type=str, default='detections.csv',
                        help='name of output file with detected keypoints')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument('--visualize', action='store_true', help='show keypoints')
    args = parser.parse_args()

    net = SinglePersonPoseEstimationWithMobileNet(num_refinement_stages=5)
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    dataset = LipValDataset(args.dataset_folder)
    evaluate(dataset, args.output_name, net, args.multiscale, args.visualize)
    pck = calc_pckh(dataset.labels_file_path, args.output_name, eval_num=len(dataset))
