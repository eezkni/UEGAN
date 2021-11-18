#-*-coding:utf-8-*-

import os
import argparse
from trainer import Trainer
from tester import Tester
from utils import create_folder, setup_seed
from config import get_config
import torch
from munch import Munch
from data_loader import get_train_loader, get_test_loader

import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main(args):
    # for fast training.
    torch.backends.cudnn.benchmark = True

    #### distributed training settings
    rank = -1
    if args.parallel_mode.lower() == 'ddp':  # disabled distributed training
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    setup_seed(args.seed)

    if rank <= 0:
        # create directories if not exist.
        create_folder(args.save_root_dir, args.version, args.model_save_path)
        create_folder(args.save_root_dir, args.version, args.sample_path)
        create_folder(args.save_root_dir, args.version, args.log_path)
        create_folder(args.save_root_dir, args.version, args.val_result_path)
        create_folder(args.save_root_dir, args.version, args.test_result_path)

    if args.mode == 'train':
        train_dataset, train_sampler=get_train_loader(root=args.train_img_dir,
                                            config=args.data_config,
                                            img_size=args.image_size,
                                            resize_size=args.resize_size,
                                            batch_size=args.train_batch_size,
                                            shuffle=args.shuffle,
                                            num_workers=args.num_workers,
                                            drop_last=args.drop_last,
                                            parallel_mode=args.parallel_mode)
        loaders = Munch(ref=train_dataset,
                        train_sampler=train_sampler,
                        val=get_test_loader(root=args.val_img_dir,
                                            config=None,
                                            dataset_type=args.data_config['val_dataset_type'],
                                            label_root=args.val_label_dir,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers),
                        qual_set=get_test_loader(root=args.qual_img_dir,
                                                config=None,
                                                label_root=args.qual_label_dir,
                                                batch_size=args.val_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers
                        ))
        trainer = Trainer(loaders, args)
        trainer.train()
    elif args.mode == 'test':
        loaders = Munch(tes=get_test_loader(root=args.test_img_dir,
                                            config=args.data_config,
                                            dataset_type=args.data_config['test_dataset_type'],
                                            label_root=args.test_label_dir,
                                            img_size=args.test_img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        tester = Tester(loaders, args)
        tester.test()
    else:
        raise NotImplementedError('Mode [{}] is not found'.format(args.mode))


if __name__ == '__main__':

    args = get_config()

    # if args.is_print_network:
    #     print(args)

    main(args)