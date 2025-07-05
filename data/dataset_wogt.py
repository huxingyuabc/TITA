import os
import random
import pathlib
import numpy as np
import torch.utils.data as data
import utils.utils_image as util

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Normalize


class MultiDataLoader:
    def __init__(self, dataloader_list):
        self.dataloaders = dataloader_list
        self.iterators = [iter(dl) for dl in self.dataloaders]
        self.lens = max([len(dataloader) for dataloader in self.dataloaders])

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for i, it in enumerate(self.iterators):
            try:
                data = next(it)
            except StopIteration:
                # re-init iterator
                it = iter(self.dataloaders[i])
                self.iterators[i] = it
                data = next(it)
            batch.append(data)
        return batch

    def reset(self):
        self.iterators = [iter(dl) for dl in self.dataloaders]

    def __len__(self):
        return self.lens


class Mixed_Dataset(Dataset):
    def __init__(self, args):
        super(Mixed_Dataset, self).__init__()

        self.args = args
        self.source1_list = []
        self.source2_list = []

        for k1,v1 in self.args['trainsets'].items():
            for k, v in v1.items():
                path1 = os.path.join(v['task_path'], v['subdir1'])
                path2 = os.path.join(v['task_path'], v['subdir2'])
                for fname in os.listdir(path1):
                    self.source1_list.append(os.path.join(path1, fname))
                    if k == 'RealMFF':
                        temp = fname.split('_')[0] + '_A.png'
                        self.source2_list.append(os.path.join(path2, temp))
                    else:
                        self.source2_list.append(os.path.join(path2, fname))

        self.transform = ToTensor()

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args['patch_size'])
        ind_W = random.randint(0, W - self.args['patch_size'])

        patch_1 = img_1[ind_H:ind_H + self.args['patch_size'], ind_W:ind_W + self.args['patch_size']]
        patch_2 = img_2[ind_H:ind_H + self.args['patch_size'], ind_W:ind_W + self.args['patch_size']]

        return patch_1, patch_2

    def __getitem__(self, idx):
        source_img1 = np.array(Image.open(self.source1_list[idx]).convert('L'))
        source_img2 = np.array(Image.open(self.source2_list[idx]).convert('L'))

        source_patch1, source_patch2 = self._crop_patch(source_img1, source_img2)

        source_patch1 = self.transform(source_patch1)
        source_patch2 = self.transform(source_patch2)

        return {'A': source_patch1, 'B': source_patch2}

    def __len__(self):
        return len(self.source1_list)


class Mixed_Equal_Dataset(Dataset):
    def __init__(self, task, trainsets, patch_size):
        super(Mixed_Equal_Dataset, self).__init__()

        self.patch_size = patch_size

        self.source1_list = []
        self.source2_list = []

        for k, v in trainsets.items():
            path1 = os.path.join(v['task_path'], v['subdir1'])
            path2 = os.path.join(v['task_path'], v['subdir2'])
            for fname in os.listdir(path1):
                self.source1_list.append(os.path.join(path1, fname))
                if k == 'RealMFF':
                    temp = fname.split('_')[0] + '_A.png'
                    self.source2_list.append(os.path.join(path2, temp))
                else:
                    self.source2_list.append(os.path.join(path2, fname))

        self.toTensor = ToTensor()
        self.batches_left = len(self.source1_list)

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.patch_size)
        ind_W = random.randint(0, W - self.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]

        return patch_1, patch_2

    def __getitem__(self, idx):
        source_img1 = np.array(Image.open(self.source1_list[idx]).convert('L'))
        source_img2 = np.array(Image.open(self.source2_list[idx]).convert('L'))

        source_patch1, source_patch2 = self._crop_patch(source_img1, source_img2)

        source_patch1 = self.toTensor(source_patch1)
        source_patch2 = self.toTensor(source_patch2)

        return {'A': source_patch1, 'B': source_patch2}

    def __len__(self):
        return len(self.source1_list)


class TestDataset(Dataset):
    def __init__(self, task_path, subdir1, subdir2):
        super(TestDataset, self).__init__()
        self.source1_list = []
        self.source2_list = []

        path1 = os.path.join(task_path, subdir1)
        path2 = os.path.join(task_path, subdir2)
        for fname in os.listdir(path1):
            self.source1_list.append(os.path.join(path1, fname))
            self.source2_list.append(os.path.join(path2, fname))

        self.toTensor = ToTensor()

    def __getitem__(self, idx):
        source1_img = np.array(Image.open(self.source1_list[idx]).convert('L'))
        source2_img = np.array(Image.open(self.source2_list[idx]).convert('L'))
        source1_rgb = np.array(Image.open(self.source1_list[idx]).convert('RGB'))
        source2_rgb = np.array(Image.open(self.source2_list[idx]).convert('RGB'))

        source1_img, source2_img = self.toTensor(source1_img), self.toTensor(source2_img)
        source1_rgb, source2_rgb = self.toTensor(source1_rgb), self.toTensor(source2_rgb)

        return {'A': source1_img, 'B': source2_img, 'A_rgb': source1_rgb, 'B_rgb': source2_rgb,
                'fname': os.path.split(self.source1_list[idx])[-1]}

    def __len__(self):
        return len(self.source1_list)
