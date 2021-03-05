from typing import Optional
import os
import random

from PIL import Image
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensor

from support_utils import yaml_parser


class LEGOdataset:
    def __init__(
            self,
            split: str = 'train',
            root: str = 'dataset',
            augments: Optional[A.Compose] = None,
            train_fraction: float = 0.8,
            shuffle: bool = True
    ) -> None:
        self._split = split
        self._root = root
        if augments:
            self._augments = augments
        else:
            self._augments = A.Compose([ToTensor()])

        if not os.path.exists(root):
            os.makedirs(root)

        if not os.listdir(root):
            source_dict = yaml_parser('./conf/conf_dataset.yaml')
            url = source_dict['dataset_url']
            zip_name = url.split('/')[-1]
            os.system("wget {} -q".format(url))
            os.system("unzip -q {} -d ./{}".format(zip_name, root))
            os.remove(zip_name)
            sys_rubish = './dataset/lego_train_images/.DS_Store'
            if os.path.exists(sys_rubish):
                os.remove(sys_rubish)

        if split in ['train', 'valid']:
            row_images = os.listdir(os.path.join(
                root, 'lego_{}_images'.format('train')))

            idx_list = list(range(len(row_images)))
            if shuffle:
                random.shuffle(idx_list)
            train_size = int(len(idx_list) * train_fraction)
            if split == 'train':
                self.images = row_images[:train_size]
            elif split == 'valid':
                self.images = row_images[train_size:]

        elif split == 'test':
            self.images = os.listdir(os.path.join(
                root, 'lego_{}_images'.format('test')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        if self._split in ['train', 'valid']:
            sub_path = 'lego_{}_images'.format('train')
            img_path = os.path.join(
                './', self._root, sub_path,
                img_name, 'images', img_name + '.png')
            img = np.array(Image.open(img_path))

            mask_path = os.path.join(self._root, sub_path, img_name, 'masks')
            mask_objs = os.listdir(mask_path)
            mask = np.zeros((img.shape[0], img.shape[1]))
            for obj in mask_objs:
                mask += np.array(Image.open(os.path.join(mask_path, obj)))

            if self._split == 'train':
                aug = self._augments(image=img, mask=mask)
                img = torch.tensor(aug['image']).permute(2, 0, 1)
                mask = torch.tensor(aug['mask'])

            elif self._split == 'valid':
                img = torch.tensor(img).permute(2, 0, 1)
                mask = torch.tensor(mask)

            img = img.float() / 255.0
            mask = mask.squeeze(0).long() // 255
            return {'image': img, 'mask': mask}

        elif self._split == 'test':
            sub_path = 'lego_{}_images'.format('test')
            img_path = os.path.join(
                './', self._root, sub_path, img_name,
                'images', img_name + '.png')
            img = np.array(Image.open(img_path))
            img = torch.tensor(img).permute(2, 0, 1)
            return {'image': img.float() / 255.0}
