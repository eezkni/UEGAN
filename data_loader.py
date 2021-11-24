#-*-coding:utf-8-*-
import random
import numpy as np
from pathlib import Path
from itertools import chain
import os
import io
import imageio
import copy
import cv2
import warnings
from numpy.core.fromnumeric import argmin

from skimage.util.dtype import img_as_int

from imresize import imresize
from munch import Munch
from PIL import Image
from utils import list_files_in_dir

import torch
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF


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
    def __init__(self, root, img_size, resize_size, config=None):
        # TODO: fix the warning with immutable Image numpy array
        warnings.filterwarnings('ignore')
        self.config = config
        self.img_size = img_size
        self.resize_size = resize_size
        self.exp_samples, self.raw_samples = self._make_dataset(root)

        self.transform = transforms.Compose([
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                       std=[0.5, 0.5, 0.5]),
                        ])

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2 = [], []
        # by sorting, exp is fnames and raw is fnames2
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            if idx == 0:
                fnames += cls_fnames
            elif idx == 1:
                fnames2 += cls_fnames

        random.shuffle(fnames)
        random.shuffle(fnames2)
        return fnames, fnames2

    def __getitem__(self, index):
        exp_fname, raw_fname = self.exp_samples[index], self.raw_samples[index]
        name = str(raw_fname)
        img_name, _ = name.split('.', 1)
        _, img_name = img_name.rsplit('/', 1)
        exp_img = Image.open(exp_fname).convert('RGB')
        raw_img = Image.open(raw_fname).convert('RGB')
        if self.transform is not None:
            exp_img = self.transform(exp_img)
            raw_img = self.transform(raw_img)
        return raw_img, exp_img, img_name, raw_img

    def __len__(self):
        return len(self.exp_samples)


class NoiseAugmentDataset(ReferenceDataset):
    def __init__(self, root, img_size, resize_size, config=None):
        super().__init__(root, img_size, resize_size, config=config)

        self.do_jpeg_aug = True if 'jpeg_aug' in self.config else False
        if self.do_jpeg_aug:
            self.jpeg_min_qual = self.config['jpeg_aug'][0]
            self.jpeg_max_qual = self.config['jpeg_aug'][1]

        self.do_scale_up_down = True if 'scale_up_down_prob' in self.config else False
        if self.do_scale_up_down:
            self.scale_up_down_prob = self.config['scale_up_down_prob']

        self.AUGMENT_TYPE = ["none", "jpg", "scale_up_down", "blur", \
                                            "jpg-scale_up_down", "jpg-blur", \
                                            "scale_up_down-blur", \
                                            "all"]
        assert len(self.config['aug_prob']) == len(self.AUGMENT_TYPE), \
                    "Not enough probs. for all augmentations"

        self.pst_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                       std=[0.5, 0.5, 0.5]),
                        ])

    def _twin_transform(self, img1, img2=None):
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
                                                img1,
                                                output_size=(self.img_size, self.img_size))
        img1 = TF.crop(img1, i, j, h, w)
        if img2 is not None:
            img2 = TF.crop(img2, i, j, h, w)

        # Random horizontal flipping
        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            if img2 is not None:
                img2 = TF.hflip(img2)

        # Random horizontal flipping
        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            if img2 is not None:
                img2 = TF.vflip(img2)

        img1 = self.pst_transform(img1)
        img2 = self.pst_transform(img2)
        return img1, img2

    def _add_scale_up_down_aug(self, img):
        #from debugpy_util import debug
        #debug(address='10.42.96.4:5678')

        h, w = img.size[:2]
        # TODO: try 2, 3 scale as well
        img = img.resize((h // 2, w // 2), Image.BICUBIC)
        img = img.resize((h, w), Image.BICUBIC)
        return img

    def _add_jpeg_aug(self, img):
        buf = io.BytesIO()
        quality = random.randint(self.jpeg_min_qual, self.jpeg_max_qual)

        # TODO: optimize it by removing imageio
        imageio.imwrite(buf, img, format='JPEG', quality=quality)
        img = imageio.imread(buf.getvalue())
        np_img = np.asarray(img)
        img = Image.fromarray(np_img)
        return img

    def _add_blur_aug(self, img):
        img = np.asarray(img)

        #TODO: tune values here
        rand_val = random.randint(0, 99)
        blur_types = ['Average', 'Bilateral', 'Box', 'Gaussian', 'Median']
        blur_type = random.choices(blur_types)[0]

        kernel_size = 3
        if rand_val % 3 == 0:
            kernel_size = 5
        if blur_type == 'Average':
            img = cv2.blur(img, (kernel_size, kernel_size))
        elif blur_type == 'Bilateral':
            img = cv2.bilateralFilter(img, 9, 75, 75)
        elif blur_type == 'Box':
            img = cv2.boxFilter(img, -1, (kernel_size, kernel_size))
        elif blur_type == 'Gaussian':
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif blur_type == 'Median':
            img = cv2.medianBlur(img, kernel_size)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        img = Image.fromarray(img)
        return img

    def _img_noise_augment(self, img):
        aug_type = random.choices(self.AUGMENT_TYPE, self.config['aug_prob'])[0]
        if aug_type == "none":
            return img
        elif aug_type == "jpg":
            img = self._add_jpeg_aug(img)
        elif aug_type == "scale_up_down":
            img = self._add_scale_up_down_aug(img)
        elif aug_type == "blur":
            img = self._add_blur_aug(img)
        elif aug_type == "jpg-scale_up_down":
            img = self._add_scale_up_down_aug(img)
            img = self._add_jpeg_aug(img)
        elif aug_type == "jpg-blur":
            img = self._add_blur_aug(img)
            img = self._add_jpeg_aug(img)
        elif aug_type == "scale_up_down-blur":
            img = self._add_scale_up_down_aug(img)
            img = self._add_blur_aug(img)
        elif "all":
            img = self._add_scale_up_down_aug(img)
            img = self._add_blur_aug(img)
            img = self._add_jpeg_aug(img)

        return img

    def _get_img(self, index):
        exp_fname, raw_fname = self.exp_samples[index], self.raw_samples[index]
        name = str(raw_fname)
        img_name, _ = name.split('.', 1)
        _, img_name = img_name.rsplit('/', 1)

        raw_img = Image.open(raw_fname).convert('RGB')
        exp_img = Image.open(exp_fname).convert('RGB')
        return exp_img, raw_img, img_name

    def __getitem__(self, index):
        exp_img, raw_img, img_name = self._get_img(index)
        orig_raw_img = raw_img

        raw_img = self._img_noise_augment(raw_img)
        # orig_raw_img.save('/tmp/img/' + img_name.split('.')[0] + '_a_pre.png')

        if self.transform is not None:
            raw_img, orig_raw_img = self._twin_transform(raw_img, orig_raw_img)
            exp_img = self.transform(exp_img)

        # orig_img_rgb = transforms.ToPILImage()(orig_raw_img)
        # orig_img_rgb.save('/tmp/img/' + img_name.split('.')[0] + '_b_orig.png')
        # img_rgb = transforms.ToPILImage()(raw_img)
        # img_rgb.save('/tmp/img/' + img_name.split('.')[0] + '_c_in.png')
        # exp_rgb = transforms.ToPILImage()(exp_img)
        # #img_rgb.save('/tmp/img/' + img_name.split('.')[0] + '_e_in.png')
        # exp_rgb.save('/tmp/img/' + img_name.split('.')[0] + '_lbl_' + str(self.exp_source_choice) + '.png')
        # print("saved")

        return raw_img, exp_img, img_name, orig_raw_img


class NonExpNoiseAugmentDataset(NoiseAugmentDataset):
    def __init__(self, root, img_size, resize_size, config=None):
        super().__init__(root, img_size, resize_size, config=config)

    def _make_dataset(self, root):
        fnames, fnames2 = [], []

        filenames = list_files_in_dir(root)
        for fn in filenames:
            file_path = os.path.join(root, fn)
            fnames.append(file_path)

        fnames2 = copy.deepcopy(fnames)
        random.shuffle(fnames)
        random.shuffle(fnames2)
        return fnames, fnames2


class MultiSourceNoiseAugmentDataset(NoiseAugmentDataset):
    def __init__(self, root, img_size, resize_size, config=None):
        super().__init__(root, img_size, resize_size, config=config)
        root = [x.strip() for x in root.split(',')]
        assert config['nb_train_datasets'] == len(root)
        raw_root = [x.strip() for x in self.config['raw_train_img_dir'].split(',')]
        assert config['raw_nb_train_datasets'] == len(raw_root)

        self.exp_total_data_sources = len(root)
        self.exp_datasets_probs = config['datasets_probs']
        self.raw_total_data_sources = len(raw_root)
        self.raw_datasets_probs = config['raw_datasets_probs']

        # TODO: Need to revist this
        smallest_exp_set = argmin(self.exp_datasets_len)
        self.dataset_len = int(self.exp_datasets_len[smallest_exp_set] / self.exp_datasets_probs[smallest_exp_set])

    def _get_img_paths(self, root):
        # if cached
        cache_filename = "_".join(root.split('/')) + '.txt'
        cache_file_path = os.path.join(self.config['cache_dir'], cache_filename)
        if os.path.isfile(cache_file_path):
            with open(cache_file_path, 'r', encoding='utf-8') as fi:
                filenames = fi.readlines()
            filenames = [x.strip() for x in filenames]
        else:
            filenames = list_files_in_dir(root)
            try:
                with open(cache_file_path, 'w', encoding='utf-8') as fi:
                    fi.write('\n'.join(filenames))
            except Exception as e:
                # clean by deleting the file
                os.remove(cache_file_path)
                print(e)
                raise e

        file_paths = []
        for fn in filenames:
            file_path = os.path.join(root, fn)
            file_paths.append(file_path)
        return file_paths

    def _read_data(self, data_paths):
        samples = []
        datasets_len = []

        for folder in data_paths:
            filenames = self._get_img_paths(folder)
            datasets_len.append(len(filenames))
            samples.append(filenames)

        return samples, datasets_len

    def _make_dataset(self, root):
        # expert data
        root = root.split(',')
        exp_samples, self.exp_datasets_len = self._read_data(root)

        # raw data
        raw_dataset_paths = self.config['raw_train_img_dir'].split(',')
        raw_samples, self.raw_datasets_len = self._read_data(raw_dataset_paths)

        return exp_samples, raw_samples

    def _get_img(self, index):
        exp_data_source_idx = random.choices(range(0, self.exp_total_data_sources),
                                weights=self.exp_datasets_probs, k=1)[0]
        raw_data_source_idx = random.choices(range(0, self.raw_total_data_sources),
                                weights=self.raw_datasets_probs, k=1)[0]

        exp_index = index % self.exp_datasets_len[exp_data_source_idx]
        exp_fname = self.exp_samples[exp_data_source_idx][exp_index]

        raw_index = index % self.raw_datasets_len[raw_data_source_idx]
        raw_fname = self.raw_samples[raw_data_source_idx][raw_index]

        name = str(raw_fname)
        img_name, _ = name.split('.', 1)
        _, img_name = img_name.rsplit('/', 1)

        raw_img = Image.open(raw_fname).convert('RGB')
        exp_img = Image.open(exp_fname).convert('RGB')

        return exp_img, raw_img, img_name

    def __len__(self):
        return self.dataset_len


class TestDataset(data.Dataset):
    def __init__(self, root, label_root, config=None):
        # TODO: fix the warning with immutable Image numpy array
        warnings.filterwarnings('ignore')
        self.samples, self.label_samples = self._make_dataset(root, label_root)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]),
            ])

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
        img = Image.fromarray(img)
        return img

    def __getitem__(self, index):
        fname_path = self.samples[index]
        img = self._get_img(fname_path)
        img = self.transform(img)
        file_name = os.path.basename(fname_path)

        # for dummy return if there is no label img
        label_img = img
        if len(self.label_samples) > 0:
            label_fname = self.label_samples[index]
            label_img = self._get_img(label_fname)
            label_img = self.transform(label_img)

        #from debugpy_util import debug
        #debug(address='10.33.72.4:5678')

        # img_rgb = transforms.ToPILImage()(img)
        # label_img_rgb = transforms.ToPILImage()(label_img)
        # img_rgb.save('/tmp/img/' + file_name.split('.')[0] + '_in.png')
        # label_img_rgb.save('/tmp/img/' + file_name.split('.')[0] + '_lbl.png')
        # print("saved")

        return img, label_img, file_name, img

    def __len__(self):
        return len(self.samples)


class TestNonExpNoiseAugmentDataset(NonExpNoiseAugmentDataset):
    '''
        Test dataset class without "expert" (label) images. Input images
        are noise augmented on the fly.
    '''
    def __init__(self, root, label_root=None, img_size=None, resize_size=None, config=None):
        super().__init__(root, img_size, resize_size, config=config)

        self.transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5]),
                        ])

    def _twin_transform(self, img1, img2=None):
        img1 = self.pst_transform(img1)
        img2 = self.pst_transform(img2)
        return img1, img2

    def _make_dataset(self, root):
        fnames, fnames2 = [], []

        filenames = list_files_in_dir(root)
        for fn in filenames:
            file_path = os.path.join(root, fn)
            fnames.append(file_path)

        fnames2 = copy.deepcopy(fnames)
        return fnames, fnames2

    def _mod_crop_img(self, img):
        img = np.asarray(img)

        H, W = img.shape[:2]
        H, W = int(H/16), int(W/16)
        # mod crop
        img = img[: H * 16, : W * 16, :]
        img = Image.fromarray(img)
        return img

    def _get_img(self, index):
        raw_fname = self.raw_samples[index]
        name = str(raw_fname)
        img_name, _ = name.split('.', 1)
        _, img_name = img_name.rsplit('/', 1)

        raw_img = Image.open(raw_fname).convert('RGB')
        raw_img = self._mod_crop_img(raw_img)

        return _, raw_img, img_name

    def __getitem__(self, index):
        # from debugpy_util import debug
        # debug('10.36.64.1:5678')
        _, raw_img, img_name = self._get_img(index)
        orig_raw_img = raw_img

        raw_img = self._img_noise_augment(raw_img)
        # orig_raw_img.save('/tmp/img/' + img_name.split('.')[0] + '_a_pre.png')

        if self.transform is not None:
            raw_img, orig_raw_img = self._twin_transform(raw_img, orig_raw_img)

        # img_rgb = transforms.ToPILImage()(raw_img)
        # img_rgb.save('/tmp/img/' + img_name.split('.')[0] + '_c_in.png')
        # orig_img_rgb = transforms.ToPILImage()(orig_raw_img)
        # exp_rgb = transforms.ToPILImage()(exp_img)
        # orig_img_rgb.save('/tmp/img/' + img_name.split('.')[0] + '_b_orig.png')
        # #img_rgb.save('/tmp/img/' + img_name.split('.')[0] + '_e_in.png')
        # exp_rgb.save('/tmp/img/' + img_name.split('.')[0] + '_lbl.png')
        # print("saved")

        return raw_img, orig_raw_img, img_name, orig_raw_img


def get_train_loader(root, config, img_size=512, \
                    resize_size=256, batch_size=8, shuffle=True, \
                    num_workers=8, drop_last=True, parallel_mode="ddp"):

    if config['dataset_type'] == "ref":
        D = ReferenceDataset
    elif config['dataset_type'] == "noise_aug":
        D = NoiseAugmentDataset
    elif config['dataset_type'] == "multi_noise_aug":
        D = MultiSourceNoiseAugmentDataset
    elif config['dataset_type'] == "non_exp_noise_aug":
        D = NonExpNoiseAugmentDataset
    elif config['dataset_type'] == "debug":
        D = DebugDataset
    else:
        raise NotImplementedError("Unrecoganized dataset type!")

    dataset = D(root, img_size, resize_size, config)

    sampler = None
    if parallel_mode == "ddp":
        world_size = torch.distributed.get_world_size()
        assert batch_size % world_size == 0
        batch_size = batch_size // world_size
        shuffle = False
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    data_loader = data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last,
                           sampler=sampler)

    return data_loader, sampler


def get_test_loader(root, config, dataset_type=None, label_root=None, \
    img_size=512, batch_size=8, shuffle=False, num_workers=4, parallel_mode="non_ddp"):

    if dataset_type is None or dataset_type == "test":
        D = TestDataset
    elif dataset_type == "test_non_exp_noise_aug":
        D = TestNonExpNoiseAugmentDataset
    else:
        raise NotImplementedError("Unrecoganized dataset type!")

    dataset = D(root, label_root=label_root, config=config)

    sampler = None
    if parallel_mode == "ddp":
        world_size = torch.distributed.get_world_size()
        assert batch_size % world_size == 0
        batch_size = batch_size // world_size
        shuffle = False
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           sampler=sampler)


class InputFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_refs(self):
        try:
            raw_img, exp_img, name, orig_raw_img = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            raw_img, exp_img, name, orig_raw_img = next(self.iter)
        return raw_img, exp_img, name, orig_raw_img

    def __next__(self):
        raw_img, exp_img, img_name, orig_raw_img = self._fetch_refs()
        raw_img, exp_img, orig_raw_img = raw_img.to(self.device), exp_img.to(self.device), orig_raw_img.to(self.device)
        inputs = Munch(img_exp=exp_img, img_raw=raw_img, img_orig_raw=orig_raw_img, img_name=img_name)

        return inputs
