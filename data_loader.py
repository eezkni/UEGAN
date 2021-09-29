#-*-coding:utf-8-*-
import numpy as np
from pathlib import Path
from itertools import chain
import os

from munch import Munch
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2 = [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            if idx == 0:
                fnames += cls_fnames
            elif idx == 1:
                fnames2 += cls_fnames
        return list(zip(fnames, fnames2))

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        name = str(fname2)
        img_name, _ = name.split('.', 1)
        _, img_name = img_name.rsplit('/', 1)
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, img_name

    def __len__(self):
        return len(self.samples)


class TestDataset(data.Dataset):
    def __init__(self, root, label_root, transform=None):
        self.samples, self.label_samples = self._make_dataset(root, label_root)
        self.transform = transform

    def _make_dataset(self, root, label_root):
        filenames = os.listdir(root)
        fnames, label_fnames = [], []
        for _, filename in enumerate(filenames):
            img_path = os.path.join(root, filename)
            fnames.append(img_path)
            if label_root is not None:
                label_img_path = os.path.join(label_root, filename)
                label_fnames.append(label_img_path)
                print(label_img_path)

        return fnames, label_fnames

    def _get_img(self, path):
        img = Image.open(path).convert('RGB')
        img = np.asarray(img)

        H, W = img.shape[:2]
        H, W = int(H/16), int(W/16)
        # mod crop
        img = img[: H * 16, : W * 16, :]

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __getitem__(self, index):
        fname_path = self.samples[index]
        img = self._get_img(fname_path)
        file_name = os.path.basename(fname_path)

        label_img = []
        if len(self.label_samples) > 0:
            label_fname = self.label_samples[index]
            label_img = self._get_img(label_fname)

        #from debugpy_util import debug
        #debug(address='10.33.72.4:5678')

        # img_rgb = transforms.ToPILImage()(img)
        # label_img_rgb = transforms.ToPILImage()(label_img)
        # img_rgb.save('/tmp/img/' + file_name.split('.')[0] + '_in.png')
        # label_img_rgb.save('/tmp/img/' + file_name.split('.')[0] + '_lbl.png')
        # print("saved")

        return label_img, img, file_name

    def __len__(self):
        return len(self.samples)


def get_train_loader(root, img_size=512, resize_size=256, batch_size=8, shuffle=True, num_workers=8, drop_last=True):

    transform = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.Resize([resize_size, resize_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ReferenceDataset(root, transform)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, label_root=None, img_size=512, batch_size=8, shuffle=False, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = TestDataset(root, label_root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_refs(self):
        try:
            x, y, name = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y, name = next(self.iter)
        return x, y, name

    def __next__(self):
        x, y, img_name = self._fetch_refs()
        x, y = x.to(self.device), y.to(self.device)
        inputs = Munch(img_exp=x, img_raw=y, img_name=img_name)

        return inputs

