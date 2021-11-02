#-*- coding:utf-8 -*-

import os
import time
import torch
import datetime
import torch.nn as nn
from torchvision.utils import save_image
from losses import PerceptualLoss, TVLoss
from utils import Logger, denorm, ImagePool, GaussianNoise
from models import Generator, Discriminator
from metrics.NIMA.CalcNIMA import calc_nima
from metrics.CalcPSNR import calc_psnr
from metrics.CalcSSIM import calc_ssim
from tqdm import *
from data_loader import InputFetcher
from utils import tensor_to_img

from torch.nn.parallel import DistributedDataParallel


class Tester(object):
    def __init__(self, loaders, args):

        # data loader
        self.loaders = loaders

        # Model configuration.
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.model_save_path = os.path.join(args.save_root_dir, args.version, args.model_save_path)
        self.sample_path = os.path.join(args.save_root_dir, args.version, args.sample_path)
        self.log_path = args.log_path
        self.test_result_path = os.path.join(args.save_root_dir, args.version, args.test_result_path)

        # Build the model and tensorboard.
        self.build_model()
        if self.args.use_tensorboard:
            self.build_tensorboard()


    def test(self):
        """ Test UEGAN ."""
        self.load_pretrained_model(self.args.pretrained_model_epoch)
        start_time = time.time()
        test_start = 0
        test_total_steps = len(self.loaders.tes)
        self.fetcher_test = InputFetcher(self.loaders.tes)

        test = {}
        test_save_path = self.test_result_path + '/' + 'test_results'
        test_compare_save_path = self.test_result_path + '/' + 'test_compare'

        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        if not os.path.exists(test_compare_save_path):
            os.makedirs(test_compare_save_path)

        self.G.eval()

        test_gen_imgs = []
        test_label_imgs = []

        pbar = tqdm(total=(test_total_steps - test_start), desc='Test epoches', position=test_start)
        pbar.write("============================== Start tesing ==============================")
        with torch.no_grad():
            for test_step in range(test_start, test_total_steps):
                input = next(self.fetcher_test)
                test_real_raw, test_real_label, test_name = input.img_raw, input.img_exp, input.img_name

                test_fake_exp = self.G(test_real_raw)

                for i in range(0, denorm(test_real_raw.data).size(0)):
                    save_imgs = denorm(test_fake_exp.data)[i:i + 1,:,:,:]

                    img_filename = os.path.basename(test_name[i]).split('.')[0]
                    save_path = os.path.join(test_save_path, '{:s}_{:0>3.2f}_testFakeExp.png'.format(img_filename, self.args.pretrained_model_epoch))
                    save_image(save_imgs, save_path)
                    if self.args.save_input:
                        label_save_img = denorm(test_real_label.data)[i:i + 1,:,:,:]
                        label_save_path = os.path.join(test_save_path, '{:s}_{:0>3.2f}_testOrig.png'.format(img_filename, self.args.pretrained_model_epoch))
                        save_image(label_save_img, label_save_path)

                        input_save_imgs = denorm(test_real_raw.data)[i:i + 1,:,:,:]
                        input_save_path = os.path.join(test_save_path, '{:s}_{:0>3.2f}_testInput.png'.format(img_filename, self.args.pretrained_model_epoch))
                        save_image(input_save_imgs, input_save_path)

                    fake_img_rgb = tensor_to_img(save_imgs.detach())
                    test_gen_imgs.append(fake_img_rgb)
                    test_real_label = denorm(test_real_label.data)[i:i + 1,:,:,:]
                    real_label_rgb = tensor_to_img(test_real_label.detach())
                    test_label_imgs.append(real_label_rgb)

                    #save_imgs_compare = torch.cat([denorm(test_real_raw.data)[i:i + 1,:,:,:], denorm(test_fake_exp.data)[i:i + 1,:,:,:]], 3)
                    #save_image(save_imgs_compare, os.path.join(test_compare_save_path, '{:s}_{:0>3.2f}_testRealRaw_testFakeExp.png'.format(img_filename, self.args.pretrained_model_epoch)))

                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                if test_step % self.args.info_step == 0:
                    pbar.write("=== Elapse:{}, Save {:>3d}-th test_fake_exp images into {} ===".format(elapsed, test_step, test_save_path))

                test['test/testFakeExp'] = denorm(test_fake_exp.detach().cpu())

                test['test_compare/testRealRaw_testFakeExp'] = torch.cat([denorm(test_real_raw.cpu()), denorm(test_fake_exp.detach().cpu())], 3)

                pbar.update(1)

                if self.args.use_tensorboard:
                    for tag, images in test.items():
                        self.logger.images_summary(tag, images, test_step + 1)

            if self.args.is_test_nima:
                self.nima_result_save_path = './results/nima_test_results/'
                curr_nima = calc_nima(test_save_path, self.nima_result_save_path,  self.args.pretrained_model_epoch)
                print("====== Avg. NIMA: {:>.4f} ======".format(curr_nima))

            if self.args.is_test_psnr_ssim:
                self.psnr_save_path = './results/psnr_test_results/'
                curr_psnr = calc_psnr(test_gen_imgs, test_label_imgs, self.psnr_save_path, self.args.pretrained_model_epoch)
                print("====== Avg. PSNR: {:>.4f} dB ======".format(curr_psnr))

                self.ssim_save_path = './results/ssim_test_results/'
                curr_ssim = calc_ssim(test_gen_imgs, test_label_imgs, self.ssim_save_path, self.args.pretrained_model_epoch)
                print("====== Avg. SSIM: {:>.4f}  ======".format(curr_ssim))


    """define some functions"""
    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.args.g_conv_dim, self.args.g_norm_fun, self.args.g_act_fun, self.args.g_use_sn).to(self.device)
        #self.D = Discriminator(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
        if self.args.parallel_mode == "dataparallel":
            self.G.to(self.args.gpu_ids[0])
            #self.D.to(self.args.gpu_ids[0])
            self.G = nn.DataParallel(self.G, self.args.gpu_ids)
            #self.D = nn.DataParallel(self.D, self.args.gpu_ids)
        #elif self.args.parallel_mode == "ddp":
        #    self.G = DistributedDataParallel(self.G, device_ids=[torch.cuda.current_device()])
        print("=== Models have been created ===")

        # print network
        if self.args.is_print_network:
            self.print_network(self.G, 'Generator')
            #self.print_network(self.D, 'Discriminator')


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("=== The number of parameters of the above model [{}] is [{}] or [{:>.4f}M] ===".format(name, num_params, num_params / 1e6))


    def load_pretrained_model(self, resume_epochs):
        checkpoint_path = os.path.join(self.model_save_path, '{}_{}_{}.pth'.format(self.args.version, self.args.adv_loss_type, resume_epochs))
        if torch.cuda.is_available():
            # save on GPU, load on GPU
            checkpoint = torch.load(checkpoint_path)
            self.G.load_state_dict(checkpoint['G_net'])
            #self.D.load_state_dict(checkpoint['D_net'])
        else:
            # save on GPU, load on CPU
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            self.G.load_state_dict(checkpoint['G_net'])
            #self.D.load_state_dict(checkpoint['D_net'])

        print("=========== loaded trained models (epochs: {})! ===========".format(resume_epochs))


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_path)