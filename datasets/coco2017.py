import math
import os
import random
import json

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset

NUM_KEYPOINTS = 17

class COCO2017TrainDataset(Dataset):

    def __init__(self, dataset_folder, stride, sigma, transform=None):
        super().__init__()
        self._dataset_folder = dataset_folder
        self._stride = stride
        self._sigma = sigma
        self._transform = transform

        annotations = json.load(open(os.path.join(dataset_folder, 
                                                 'annotations', 
                                                 'person_keypoints_train2017.json')))
        self.annotations = annotations['annotations']

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = str(ann['image_id']).zfill(12) + '.jpg'
        image = cv2.imread(os.path.join(self._dataset_folder, 'train2017', img_path), cv2.IMREAD_COLOR)
        h, w, c = image.shape
        if random.random() > 0.5:
            center_x = random.randint(w//3, w-1-w//3)
            center_y = random.randint(h//3, h-1-h//3)
            percentage = 0.3
            kpt = [
                [center_x - random.randint(1, int(w*percentage)), center_y - random.randint(1, int(h*percentage))],
                [center_x + random.randint(1, int(w*percentage)), center_y - random.randint(1, int(h*percentage))],
                [center_x + random.randint(1, int(w*percentage)), center_y + random.randint(1, int(h*percentage))],
                [center_x - random.randint(1, int(w*percentage)), center_y + random.randint(1, int(h*percentage))]
                ]
            cv2.fillConvexPoly(image, np.array(kpt, dtype=np.int32), (128, 128, 128))

        keypoints = np.asarray(ann['keypoints'], dtype=np.float32)

        sample = {
            'keypoints': keypoints,
            'image': image,
        }
        if self._transform:
            sample = self._transform(sample)

        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps

        image = sample['image'].astype(np.float32)
        image = (image - 128) / 256
        sample['image'] = image.transpose((2, 0, 1))
        return sample

    def __len__(self):
        return len(self.annotations)

    def _generate_keypoint_maps(self, sample):
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(NUM_KEYPOINTS + 1,
                                        n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg

        keypoints = sample['keypoints']
        for id in range(len(keypoints) // 3):
            if keypoints[id * 3 + 2] == 0:
                continue
            self._add_gaussian(keypoint_maps[id], keypoints[id * 3], keypoints[id * 3 + 1], self._stride, self._sigma)
        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)

        return keypoint_maps

    def _add_gaussian(self, keypoint_map, x, y, stride, sigma):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                     (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1


class COCO2017ValDataset(Dataset):

    def __init__(self, dataset_folder, num_images=-1):
        super().__init__()
        self._dataset_folder = dataset_folder
        annotations = json.load(open(os.path.join(dataset_folder, 
                                                 'annotations', 
                                                 'person_keypoints_val2017.json')))
        self.annotations = annotations['annotations']
        if num_images > 0:
            self.annotations = self.annotations[:num_images]

    def __getitem__(self, idx):
        img_path = str(self.annotations[idx])['image_id'].zfill(12) + '.jpg'
        image = cv2.imread(os.path.join(self._dataset_folder, 'val2017', img_path), cv2.IMREAD_COLOR)
        sample = {
            'image': image,
            'file_name': img_path
        }
        return sample

    def __len__(self):
        return len(self.annotations)
