#-*-coding:utf-8-*-
import random
import numpy as np
from pathlib import Path
from itertools import chain
import os
import io
import imageio

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
    def __init__(self, root, config=None, transform=None):
        self.samples = self._make_dataset(root)
        self.transform = transform
        self.config = config

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

        random.shuffle(fnames)
        random.shuffle(fnames2)

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


class NoiseAugmentDataset(ReferenceDataset):
    def __init__(self, root, config, transform=None):
        super().__init__(root, transform=transform, config=config)
        self.do_jpeg_aug = True if 'jpeg_aug' in self.config else False
        self.jpeg_min_qual = self.config['jpeg_aug'][0]
        self.jpeg_max_qual = self.config['jpeg_aug'][1]

    def _add_jpeg_aug(self, img):
        if self.do_jpeg_aug and self.config['jpeg_prob'] > random.uniform(0, 1):
            buf = io.BytesIO()
            quality = random.randint(self.jpeg_min_qual, self.jpeg_max_qual)

            # TODO: optimize it by removing imageio
            imageio.imwrite(buf, img, format='JPEG', quality=quality)
            img = imageio.imread(buf.getvalue())
            np_img = np.asarray(img)
            img = Image.fromarray(np_img)

        return img

    def __getitem__(self, index):
        exp_fname, raw_fname = self.samples[index]
        name = str(raw_fname)
        img_name, _ = name.split('.', 1)
        _, img_name = img_name.rsplit('/', 1)

        raw_img = Image.open(raw_fname).convert('RGB')
        exp_img = Image.open(exp_fname).convert('RGB')

        #raw_img.save('/tmp/img/' + img_name.split('.')[0] + '_apre.png')

        # jpeg noise augmentation
        #from debugpy_util import debug
        #debug(address='10.42.96.4:5678')
        if  self.do_jpeg_aug:
            raw_img = self._add_jpeg_aug(raw_img)

        if self.transform is not None:
            raw_img = self.transform(raw_img)
            exp_img = self.transform(exp_img)

        '''
            img_rgb = transforms.ToPILImage()(raw_img)
            exp_rgb = transforms.ToPILImage()(exp_img)
            img_rgb.save('/tmp/img/' + img_name.split('.')[0] + '_bin.png')
            exp_rgb.save('/tmp/img/' + img_name.split('.')[0] + '_lbl.png')
            print("saved")
        '''
        return exp_img, raw_img, img_name


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

        # for dummy return if there is no label img
        label_img = img
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


def get_train_loader(root, config, img_size=512, \
                    resize_size=256, batch_size=8, shuffle=True, \
                    num_workers=8, drop_last=True):

    transform = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.Resize([resize_size, resize_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if config['dataset_type'] == "ref":
        D = ReferenceDataset
    elif config['dataset_type'] == "noise_aug":
        D = NoiseAugmentDataset
    else:
        raise NotImplementedError("Unrecoganized dataset type!")

    dataset = D(root, config, transform)

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

