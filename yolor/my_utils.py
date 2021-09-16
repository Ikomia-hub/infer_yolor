import torch
import os
from torch.utils.data import Dataset
import cv2
import glob
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywh2xyxy
from utils.torch_utils import torch_distributed_zero_first
import random
from utils.datasets import random_perspective, augment_hsv, letterbox
import matplotlib.pyplot as plt

def create_dataloader(ikDataset, imgsz, batch_size, stride, single_cls, hyp=None, augment=False, cache=False,
                      pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(ikDataset, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      rank=rank)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None

    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=nw,
                                    sampler=sampler,
                                    pin_memory=True,
                                    collate_fn=LoadImagesAndLabels.collate_fn)  # torch.utils.data.DataLoader()

    return dataloader, dataset


def split_train_test(ikDataset, ratio_split_train_test, seed=0):
    random.seed(seed)
    annoted_imgs = ikDataset['images']
    random.shuffle(annoted_imgs)
    pivot = int((len(annoted_imgs) - 1) * ratio_split_train_test)
    return {'images':annoted_imgs[:pivot],'metadata':ikDataset['metadata']}, {'images':annoted_imgs[pivot:],'metadata':ikDataset['metadata']}
    # (train_data, test_data)


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class My_dataset(Dataset):
    def __init__(self, ikDataset, transform=None):
        self.transform = transform  # using transform in torch!
        self.sample_list = ikDataset

    def __len__(self):
        return len(self.sample_list["images"])

    def __getitem__(self, idx):

        record = self.sample_list["images"][idx]
        if "annotations" in record:
            image, bboxes = cv2.imread(record["filename"]), record["annotations"]

            sample = {'images': image, 'bboxes': bboxes}
            if self.transform:
                sample = self.transform(sample)
        else:
            return None
        return sample


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, ikDataset, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 4, -img_size // 4]
        self.stride = stride
        self.dataset = ikDataset
        self.labels = [[[a['category_id']] + a['bbox'] for a in list_dict] for list_dict in
                       [dict['annotations'] for dict in self.dataset['images']]]


    def load_mosaic(self, index):
        # loads images in a mosaic

        labels4 = []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.dataset['images']) - 1) for _ in
                             range(3)]  # 3 additional image indices

        for i, index in enumerate(indices):
            # Load image
            img = cv2.imread(self.dataset['images'][index]['filename'])
            img = cv2.resize(img, (s,s), interpolation = cv2.INTER_AREA)
            h, w = self.dataset['images'][index]['height'],self.dataset['images'][index]['width']

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - s, 0), max(yc - s, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = s - (x2a - x1a), s - (y2a - y1a), s, s  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - s, 0), min(xc + s, s * 2), yc
                x1b, y1b, x2b, y2b = 0, s - (y2a - y1a), min(s, x2a - x1a), s
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - s, 0), yc, xc, min(s * 2, yc + s)
                x1b, y1b, x2b, y2b = s - (x2a - x1a), 0, s, min(y2a - y1a, s)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + s, s * 2), min(s * 2, yc + s)
                x1b, y1b, x2b, y2b = 0, 0, min(s, x2a - x1a), min(y2a - y1a, s)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            x = np.array(self.labels[index])
            labels = x.copy()

            if x.size > 0:  # xywh to pixel xyxy format
                labels[:, 1] = s/w * x[:, 1] + padw
                labels[:, 2] = s/h * x[:, 2] + padh
                labels[:, 3] = s/w * (x[:, 1] + x[:, 3] ) + padw
                labels[:, 4] = s/h * (x[:, 2] + x[:, 4] ) + padh

            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_perspective
            # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4 = random_perspective(img4, labels4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        # To visualize input
        """
        for l in labels4:
            img4 = self.draw_rectangle(img4,*l[1:])
        cv2.imwrite("mosaique_perspective.png",img4)
        """
        return img4, labels4

    def __len__(self):
        return len(self.dataset['images'])

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            # img, labels = load_mosaic(self, index)
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                # img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                img2, labels2 = self.load_mosaic(random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img = cv2.imread(self.dataset['images'][index]['filename'])
            h,w,_ = np.shape(img)
            img = cv2.resize(img, (self.img_size,self.img_size), interpolation = cv2.INTER_AREA)
            shapes = (h, w), ((self.img_size / h, self.img_size / w), 0)  # for COCO mAP rescaling
            # Load labels
            labels = []
            x = np.array(self.labels[index])
            if x.size > 0:
                labels = x.copy()
                # Normalized xywh to pixel xyxy format
                labels[:, 1] = self.img_size / w * x[:, 1]
                labels[:, 2] = self.img_size / h * x[:, 2]
                labels[:, 3] = self.img_size / w * (x[:, 1] + x[:, 3])
                labels[:, 4] = self.img_size / h * (x[:, 2] + x[:, 4])

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= self.img_size  # normalized height 0-1
            labels[:, [1, 3]] /= self.img_size  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]


        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.dataset['images'][index]['filename'], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
