#-*- coding:utf-8 -*-

from logging import raiseExceptions
import os
import time
import torch
import datetime
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from losses import PerceptualLoss, GANLoss, MultiscaleRecLoss
from utils import Logger, denorm, ImagePool, tensor_to_img, tensor_bgr_to_gray_scale
from models import Generator, Discriminator
from metrics.NIMA.CalcNIMA import calc_nima
from metrics.CalcPSNR import calc_psnr
from metrics.CalcSSIM import calc_ssim
from tqdm import *
from data_loader import InputFetcher
from torch.nn.parallel import DistributedDataParallel


class Trainer(object):
    def __init__(self, loaders, args):
        # data loader
        self.loaders = loaders

        # Model configuration.
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.parallel_mode = self.args.parallel_mode

        self.rank = -1
        if self.parallel_mode == "ddp":
            self.rank = torch.distributed.get_rank()

        # Directories.
        self.model_save_path = os.path.join(args.save_root_dir, args.version, args.model_save_path)
        self.sample_path = os.path.join(args.save_root_dir, args.version, args.sample_path)
        self.log_path = os.path.join(args.log_path, args.version)
        self.val_result_path = os.path.join(args.save_root_dir, args.version, args.val_result_path)
        self.lr_G = args.g_lr
        self.lr_D = args.d_lr

        # Build the model and tensorboard.
        self.criterionIdt = MultiscaleRecLoss(scale=3, rec_loss_type=self.args.idt_loss_type, multiscale=True, loss_wts=self.args.idt_loss_wts)
        self.build_model()
        if self.args.use_tensorboard:
            self.build_tensorboard()

        self.count_until_log = 0
        self.loss = {}
        self.d_metric = {}
        self.loss['D/Total'] = 0.
        self.loss['G/Total'] = 0.
        self.loss['G/adv_loss'] = 0.
        self.loss['G/percep_loss'] = 0.
        self.loss['G/idt_loss'] = 0.

        self.d_metric['D/real'] = 0.
        self.d_metric['D/fake'] = 0.

    def train(self):
        """ Train UEGAN ."""
        self.fetcher = InputFetcher(self.loaders.ref)
        self.fetcher_val = InputFetcher(self.loaders.val)
        self.fetcher_quality = InputFetcher(self.loaders.qual_set)

        self.train_steps_per_epoch = len(self.loaders.ref)

        # set nima, psnr, ssim global parameters
        if self.args.is_test_nima:
            self.best_nima_epoch, self.best_nima = 0, 0.0
        if self.args.is_test_psnr_ssim:
            self.best_psnr_epoch, self.best_psnr = 0, 0.0
            self.best_ssim_epoch, self.best_ssim = 0, 0.0

        # set loss functions
        self.criterionPercep = PerceptualLoss()
        self.criterionPercep.to(self.device)
        if self.args.parallel_mode.lower() == "dataparallel":
            self.criterionPercep = nn.DataParallel(self.criterionPercep, self.args.gpu_ids)

        # TODO: check if DDP speeds up for feature loss
        # elif self.args.parallel_mode.lower() == "ddp":
        #     self.criterionPercep = DistributedDataParallel(self.criterionPercep,
        #                                             device_ids=[torch.cuda.current_device()])

        self.criterionGAN = GANLoss(self.args.adv_loss_type, tensor=torch.cuda.FloatTensor)

        start_step = 0
        self.epoch = 0

        # Model loading can be either from a path or epoch number
        if self.args.pretrained_model_epoch is not None and self.args.pretrained_model_path is not None:
            raiseExceptions("pretrained_model_epoch and pretrained_model_path are exclusive")
        # start from scratch or trained models
        elif self.args.pretrained_model_epoch is not None:
            # resume training
            start_step = int(self.args.pretrained_model_epoch * self.train_steps_per_epoch)
            self.load_pretrained_model_resume_training(self.args.pretrained_model_epoch)
            self.epoch += 1
        elif self.args.pretrained_model_path is not None:
            self.load_pretrained_model(self.args.pretrained_model_path)

        # start training
        if self.is_main_process():
            print("======================================= start training =======================================")

        self.start_time = time.time()
        total_steps = int(self.args.total_epochs * self.train_steps_per_epoch)
        self.val_start_steps = int(self.args.num_epochs_start_val * self.train_steps_per_epoch)
        self.val_each_steps = int(self.args.val_interval_rel_epoch * self.train_steps_per_epoch)
        self.model_save_step = int(self.args.model_save_interval * self.train_steps_per_epoch)

        if self.is_main_process():
            print("=========== start to iteratively train generator and discriminator ===========")
            pbar = tqdm(total=total_steps, desc='Train epoches', initial=start_step)

        for step in range(start_step, total_steps):
            if self.parallel_mode == "ddp" and step % self.train_steps_per_epoch == 0:
                if self.is_main_process(): print("============= Sampler epoch updated ======================")
                self.loaders.train_sampler.set_epoch(self.epoch)

            ########## model train
            self.G.train()
            self.D.train()

            ########## data iter
            input = next(self.fetcher)
            self.real_raw = input.img_raw
            self.real_exp = input.img_exp
            self.orig_raw = input.img_orig_raw
            self.real_raw_name = input.img_name

            ########## forward
            self.fake_exp = self.G(self.real_raw)
            self.fake_exp_store = self.fake_exp_pool.query(self.fake_exp)

            ########## update D
            for p in self.D.parameters():
                p.requires_grad = True

            self.d_optimizer.zero_grad()
            real_exp_preds = self.D(self.real_exp)
            fake_exp_preds = self.D(self.fake_exp_store.detach())
            d_loss = self.criterionGAN(real_exp_preds, fake_exp_preds, None, None, for_discriminator=True)
            if self.args.adv_input:
                input_preds = self.D(self.real_raw)
                d_loss += self.criterionGAN(real_exp_preds, input_preds, None, None, for_discriminator=True)
            d_loss.backward()
            self.d_optimizer.step()
            self.d_loss = d_loss.item()

            # calculate accuracy of D
            self.d_real_acc_proxy = 0.
            self.d_fake_acc_proxy = 0.
            for (pred_real_i, pred_fake_i) in zip(real_exp_preds, fake_exp_preds):
                if isinstance(pred_real_i, list):
                    pred_real_i = pred_real_i[-1]
                if isinstance(pred_fake_i, list):
                    pred_fake_i = pred_fake_i[-1]

                self.d_real_acc_proxy += torch.mean(pred_real_i.detach())
                self.d_fake_acc_proxy += torch.mean(pred_fake_i.detach())

            ########## update G
            # Freeze D
            for p in self.D.parameters():
                p.requires_grad = False

            self.g_optimizer.zero_grad()
            real_exp_preds = self.D(self.real_exp)
            fake_exp_preds = self.D(self.fake_exp)
            g_adv_loss = self.args.lambda_adv * self.criterionGAN(real_exp_preds, fake_exp_preds, None, None, for_discriminator=False)
            self.g_adv_loss = g_adv_loss.item()
            g_loss = g_adv_loss

            if self.args.black_n_white_loss:
                fake_denom = (self.fake_exp+1.)/2.
                fake_grayscale = tensor_bgr_to_gray_scale(fake_denom)

                orig_denom = (self.orig_raw+1.)/2.
                orig_grayscale = tensor_bgr_to_gray_scale(orig_denom)

                g_percep_loss = self.args.lambda_percep * self.criterionPercep(fake_grayscale, orig_grayscale)
            else:
                g_percep_loss = self.args.lambda_percep * self.criterionPercep((self.fake_exp+1.)/2., (self.orig_raw+1.)/2.)

            if self.args.parallel_mode.lower() == "dataparallel":
                g_percep_loss = g_percep_loss.mean()

            self.g_percep_loss = g_percep_loss.item()
            g_loss += g_percep_loss

            self.real_exp_idt = self.G(self.real_exp)
            g_idt_loss = self.args.lambda_idt * self.criterionIdt(self.real_exp_idt, self.real_exp)
            self.g_idt_loss = g_idt_loss.item()
            g_loss += g_idt_loss

            g_loss.backward()
            self.g_optimizer.step()
            self.g_loss = g_loss.item()

            self.lr_G = self.g_optimizer.param_groups[0]['lr']
            self.lr_D = self.d_optimizer.param_groups[0]['lr']

            if self.is_main_process():
                ### print info and save models
                self.print_info(step, total_steps, pbar)

                ### logging using tensorboard
                self.logging(step)

                ### validation
                self.model_validation(step, self.fetcher_val, len(self.loaders.val))
                self.model_validation(step, self.fetcher_quality, len(self.loaders.qual_set), save_imgs=True, cal_metrics=False)

            ### learning rate update
            ### learning rate update
            if (step + 1) % self.train_steps_per_epoch == 0:
                current_epoch = step // self.train_steps_per_epoch
                if self.lr_G > self.args.min_lr_g and current_epoch > self.args.lr_num_epochs_decay:
                    self.lr_scheduler_g.step()
                if self.lr_D > self.args.min_lr_d and current_epoch > self.args.lr_num_epochs_decay:
                    self.lr_scheduler_d.step()
                if self.is_main_process():
                    for param_group in self.g_optimizer.param_groups:
                        pbar.write("====== Epoch: {:>3d}/{}, Learning rate(lr) of Encoder(E) and Generator(G): [{}], ".format(((step + 1) // self.train_steps_per_epoch), self.args.total_epochs, param_group['lr']), end='')
                    for param_group in self.d_optimizer.param_groups:
                        pbar.write("Learning rate (lr) of Discriminator(D): [{}] ======".format(param_group['lr']))

                # update epoch
                self.epoch += 1

            if self.is_main_process():
                pbar.update(1)
                pbar.set_description(f"Train epoch %.2f" % ((step+1.0)/self.train_steps_per_epoch))

        if self.is_main_process():
            self.val_best_results()

            pbar.write("=========== Complete training ===========")
            pbar.close()

    def is_main_process(self):
        if self.rank <= 0:
            return True
        return False

    def logging(self, step):
        self.count_until_log += 1
        self.loss['D/Total'] += self.d_loss
        self.loss['G/Total'] += self.g_loss
        self.loss['G/adv_loss'] += self.g_adv_loss
        self.loss['G/percep_loss'] += self.g_percep_loss
        self.loss['G/idt_loss'] += self.g_idt_loss

        self.d_metric['D/real'] += self.d_real_acc_proxy
        self.d_metric['D/fake'] += self.d_fake_acc_proxy

        if (step+1) % self.args.log_step == 0:
            self.lr = {}
            self.lr['lr/G'] = self.lr_G
            self.lr['lr/D'] = self.lr_D
            if self.args.use_tensorboard:
                for tag, value in self.loss.items():
                    mean_val = value / self.count_until_log
                    self.logger.scalar_summary(tag, mean_val, step+1)
                for tag, value in self.lr.items():
                    self.logger.scalar_summary(tag, value, step+1)
                for tag, value in self.d_metric.items():
                    mean_val = value / self.count_until_log
                    self.logger.scalar_summary(tag, value, step+1)

            self.count_until_log = 0
            self.loss = {}
            self.loss['D/Total'] = 0.
            self.loss['G/Total'] = 0.
            self.loss['G/adv_loss'] = 0.
            self.loss['G/percep_loss'] = 0.
            self.loss['G/idt_loss'] = 0.

            self.d_metric['D/real'] = 0.
            self.d_metric['D/fake'] = 0.

        if (step+1) % self.args.img_log_step == 0:
            self.images = {}
            self.images['Train_realExpIdt/realExp_realExpIdt'] = torch.cat([denorm(self.real_exp.cpu()), denorm(self.real_exp_idt.detach().cpu())], 3)
            self.images['Train_compare/realRaw_fakeExp_realExp'] = torch.cat([denorm(self.real_raw.cpu()), denorm(self.fake_exp.detach().cpu()), denorm(self.real_exp.cpu())], 3)
            self.images['Train_fakeExp/fakeExp'] = denorm(self.fake_exp.detach().cpu())
            self.images['Train_fakeExpStore/fakeExpStore'] = denorm(self.fake_exp_store.detach().cpu())

            for tag, image in self.images.items():
                self.logger.images_summary(tag, image, step+1)


    def print_info(self, step, total_steps, pbar):
        current_epoch = self.epoch

        if (step + 1) % self.args.info_step == 0:
            elapsed_num = time.time() - self.start_time
            elapsed = str(datetime.timedelta(seconds=elapsed_num))
            pbar.write("Elapse:{:>.12s}, D_Step:{:>6d}/{}, G_Step:{:>6d}/{}, D_loss:{:>.4f}, G_loss:{:>.4f}, G_percep_loss:{:>.4f}, G_adv_loss:{:>.4f}, G_idt_loss:{:>.4f}".format(elapsed, step + 1, total_steps, (step + 1), total_steps, self.d_loss, self.g_loss, self.g_percep_loss, self.g_adv_loss, self.g_idt_loss))

        # sample images
        if (step + 1) % self.args.sample_step == 0:
            for i in range(0, self.real_raw.size(0)):
                save_imgs = torch.cat([denorm(self.real_raw.data)[i:i + 1,:,:,:], denorm(self.fake_exp.data)[i:i + 1,:,:,:], denorm(self.real_exp.data)[i:i + 1,:,:,:]], 3)
                save_image(save_imgs, os.path.join(self.sample_path, '{:s}_{:0>3.2f}_{:0>2d}_realRaw_fakeExp_realExp.png'.format(self.real_raw_name[i], current_epoch, i)))

        # save models
        if (step + 1) % self.model_save_step == 0:
            if self.parallel_mode in ["dataparallel", "ddp"]:
                if torch.cuda.device_count() > 1:
                    checkpoint = {
                    "G_net": self.G.module.state_dict(),
                    "D_net": self.D.module.state_dict(),
                    "epoch": current_epoch,
                    "g_optimizer": self.g_optimizer.state_dict(),
                    "d_optimizer": self.d_optimizer.state_dict(),
                    "lr_scheduler_g": self.lr_scheduler_g.state_dict(),
                    "lr_scheduler_d": self.lr_scheduler_d.state_dict()
                    }
                else:
                    raise NotImplementedError('Model saving condition fell in a crack')
            else:
                checkpoint = {
                    "G_net": self.G.state_dict(),
                    "D_net": self.D.state_dict(),
                    "epoch": current_epoch,
                    "g_optimizer": self.g_optimizer.state_dict(),
                    "d_optimizer": self.d_optimizer.state_dict(),
                    "lr_scheduler_g": self.lr_scheduler_g.state_dict(),
                    "lr_scheduler_d": self.lr_scheduler_d.state_dict()
                }
            torch.save(checkpoint, os.path.join(self.model_save_path, '{}_{}_{}.pth'.format(self.args.version, self.args.adv_loss_type, step)))

            pbar.write("======= Save model checkpoints into {} ======".format(self.model_save_path))


    def model_validation(self, step, data_fetcher, data_size, save_imgs=False, cal_metrics=True):
        if (step + 1) >= self.val_start_steps:
            if (step + 1) % self.val_each_steps == 0:
                val = {}
                current_epoch = step
                val_start = 0
                val_total_steps = data_size


                self.G.eval()

                pbar = tqdm(total=(val_total_steps - val_start), desc='Validation epoches', position=val_start)
                pbar.write("============================== Start validation ==============================")

                if save_imgs:
                    print("Saving imgs")

                val_gen_imgs = []
                val_label_imgs = []

                with torch.no_grad():
                    for val_step in range(val_start, val_total_steps):

                        input = next(data_fetcher)
                        val_real_raw, val_real_label, _, val_name = input.img_raw, input.img_exp, input.img_orig_raw, input.img_name
                        val_fake_exp = self.G(val_real_raw)

                        for i in range(0, denorm(val_real_raw.data).size(0)):
                            fake_save_imgs = denorm(val_fake_exp.data)[i:i + 1,:,:,:]

                            if save_imgs:
                                img_name = val_name[i].split('.')[0]
                                val_save_path = self.val_result_path + '/' + 'out_imgs/' + img_name
                                #val_compare_save_path = self.val_result_path + '/' + 'out_' + str(current_epoch)

                                if not os.path.exists(val_save_path):
                                    os.makedirs(val_save_path)

                                file_save_path = os.path.join(val_save_path, '{:s}_{}_valFakeExp.png'.format(img_name, int(current_epoch)))
                                save_image(fake_save_imgs, file_save_path)

                            fake_img_rgb = tensor_to_img(fake_save_imgs.detach())
                            val_gen_imgs.append(fake_img_rgb)
                            val_real_label = denorm(val_real_label.data)[i:i + 1,:,:,:]
                            real_label_rgb = tensor_to_img(val_real_label.detach())
                            val_label_imgs.append(real_label_rgb)

                            #save_imgs_compare = torch.cat([denorm(val_real_raw.data)[i:i + 1,:,:,:], denorm(val_fake_exp.data)[i:i + 1,:,:,:]], 3)
                            #save_image(save_imgs_compare, os.path.join(val_compare_save_path, '{:s}_{:0>3.2f}_valRealRaw_valFakeExp.png'.format(val_name[i], current_epoch)))

                        elapsed = time.time() - self.start_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        if val_step % self.args.info_step == 0:
                            pbar.write("=== Elapse:{}, Save {:>3d}-th val_fake_exp images ===".format(elapsed, val_step))

                        val['val/valFakeExp'] = denorm(val_fake_exp.detach().cpu())

                        val['val_compare/valRealRaw_valFakeExp'] = torch.cat([denorm(val_real_raw.cpu()), denorm(val_fake_exp.detach().cpu())], 3)

                        pbar.update(1)

                        if self.args.use_tensorboard:
                            for tag, images in val.items():
                                self.logger.images_summary(tag, images, val_step + 1)

                    pbar.close()
                    if cal_metrics:
                        if self.args.is_test_nima:
                            self.nima_result_save_path = os.path.join(self.args.save_root_dir, self.args.version, 'results/nima_val_results/')
                            curr_nima = calc_nima(val_save_path, self.nima_result_save_path,  current_epoch)
                            if self.best_nima < curr_nima:
                                self.best_nima = curr_nima
                                self.best_nima_epoch = current_epoch
                            print("====== Avg. NIMA: {:>.4f} ======".format(curr_nima))

                        if self.args.is_test_psnr_ssim:
                            self.psnr_save_path = os.path.join(self.args.save_root_dir, self.args.version, 'results/psnr_val_results/')
                            curr_psnr = calc_psnr(val_gen_imgs, val_label_imgs, self.psnr_save_path, current_epoch)
                            if self.best_psnr < curr_psnr:
                                self.best_psnr = curr_psnr
                                self.best_psnr_epoch = current_epoch
                            print("====== Avg. PSNR: {:>.4f} dB ======".format(curr_psnr))

                            self.ssim_save_path = os.path.join(self.args.save_root_dir, self.args.version, 'results/ssim_val_results/')
                            curr_ssim = calc_ssim(val_gen_imgs, val_label_imgs, self.ssim_save_path, current_epoch)
                            if self.best_ssim < curr_ssim:
                                self.best_ssim = curr_ssim
                                self.best_ssim_epoch = current_epoch
                            print("====== Avg. SSIM: {:>.4f}  ======".format(curr_ssim))
                torch.cuda.empty_cache()
                time.sleep(2)


    def val_best_results(self):
        if self.args.is_test_psnr_ssim:
            if not os.path.exists(self.psnr_save_path):
                os.makedirs(self.psnr_save_path)
            psnr_result = self.psnr_save_path + 'PSNR_total_results_epoch_avgpsnr.csv'
            psnrfile = open(psnr_result, 'a+')
            psnrfile.write('Best epoch: ' + str(self.best_psnr_epoch) + ',' + str(round(self.best_psnr, 6)) + '\n')
            psnrfile.close()

            if not os.path.exists(self.ssim_save_path):
                os.makedirs(self.ssim_save_path)
            ssim_result = self.ssim_save_path + 'SSIM_total_results_epoch_avgssim.csv'
            ssimfile = open(ssim_result, 'a+')
            ssimfile.write('Best epoch: ' + str(self.best_ssim_epoch) + ',' + str(round(self.best_ssim, 6)) + '\n')
            ssimfile.close()

        if self.args.is_test_nima:
            nima_total_result = self.nima_result_save_path + 'NIMA_total_results_epoch_mean_std.csv'
            totalfile = open(nima_total_result, 'a+')
            totalfile.write('Best epoch:' + str(self.best_nima_epoch) + ',' + str(round(self.best_nima, 6)) + '\n')
            totalfile.close()


    """define some functions"""
    def build_model(self):
        """Create a generator and a discriminator."""
        # G = Generator
        # if self.args.big_g:
        #     G = BigGenerator
        self.G = Generator(self.args.g_conv_dim, self.args.g_norm_fun, self.args.g_act_fun, self.args.g_use_sn).to(self.device)
        self.D = Discriminator(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)

        self.G.to(self.device)
        self.D.to(self.device)
        if self.args.parallel_mode.lower() == "dataparallel":
            self.G = nn.DataParallel(self.G, self.args.gpu_ids)
            self.D = nn.DataParallel(self.D, self.args.gpu_ids)
        elif self.args.parallel_mode.lower() == "ddp":
            self.G = DistributedDataParallel(self.G, device_ids=[torch.cuda.current_device()])
            self.D = DistributedDataParallel(self.D, device_ids=[torch.cuda.current_device()])

        if self.is_main_process():
            print("=== Models have been created ===")

        if self.is_main_process() and self.args.is_print_network:
            # print network
            self.print_network(self.G, 'Generator')
            self.print_network(self.D, 'Discriminator')

        # init network
        if self.args.init_type:
            self.init_weights(self.G, init_type=self.args.init_type, gain=0.02)
            self.init_weights(self.D, init_type=self.args.init_type, gain=0.02)

        # optimizer
        if self.args.optimizer_type == 'adam':
            # Adam optimizer
            self.g_optimizer = torch.optim.Adam(params=self.G.parameters(), lr=self.args.g_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            self.d_optimizer = torch.optim.Adam(params=self.D.parameters(), lr=self.args.d_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
        elif self.args.optimizer_type == 'rmsprop':
            # RMSprop optimizer
            self.g_optimizer = torch.optim.RMSprop(params=self.G.parameters(), lr=self.args.g_lr, alpha=self.args.alpha)
            self.d_optimizer = torch.optim.RMSprop(params=self.D.parameters(), lr=self.args.d_lr, alpha=self.args.alpha)
        else:
            raise NotImplementedError("=== Optimizer [{}] is not found ===".format(self.args.optimizer_type))

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=self.args.lr_decay_ratio)
            self.lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer, gamma=self.args.lr_decay_ratio)
            if self.is_main_process(): print("=== Set learning rate decay policy for Generator(G) and Discriminator(D) ===")

        self.fake_exp_pool = ImagePool(self.args.pool_size)


    def init_weights(self, net, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('Initialization method [{}] is not implemented'.format(init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data,   0.0)
        if self.is_main_process(): print("=== Initialize network with [{}] ===".format(init_type))
        net.apply(init_func)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        trainable_num_params = 0
        for p in model.parameters():
            num_params += p.numel()
            if p.requires_grad:
                trainable_num_params += p.numel()
        # print(model)
        print("=== The number of parameters of the above model [{}] is [{}] or [{:>.4f}M] ===".format(name, num_params, num_params / 1e6))
        print("=== The no of trainable_params of the above model [{}] is [{}] or [{:>.4f}M] ===".format(name, num_params, num_params / 1e6))


    def load_pretrained_model_resume_training(self, resume_epochs):
        checkpoint_path = os.path.join(self.model_save_path, '{}_{}_{}.pth'.format(self.args.version, self.args.adv_loss_type, resume_epochs))
        if torch.cuda.is_available():
            # save on GPU, load on GPU
            checkpoint = torch.load(checkpoint_path)
            self.G.load_state_dict(checkpoint['G_net'])
            self.D.load_state_dict(checkpoint['D_net'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g'])
            self.lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d'])
        else:
            # save on GPU, load on CPU
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            self.G.load_state_dict(checkpoint['G_net'])
            self.D.load_state_dict(checkpoint['D_net'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g'])
            self.lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d'])

        self.epoch = checkpoint['epoch']

        print("=========== loaded trained models (epochs: {})! ===========".format(resume_epochs))

    def load_pretrained_model(self, checkpoint_path):
        if torch.cuda.is_available():
            # save on GPU, load on GPU
            checkpoint = torch.load(checkpoint_path)

            # if distributed, need to append "module"
            if torch.distributed.is_initialized():
                G_state_dict = {}
                for key, value in checkpoint['G_net'].items():
                    key = 'module.' + key
                    G_state_dict[key] = value
                self.G.load_state_dict(G_state_dict)

                D_state_dict = {}
                for key, value in checkpoint['D_net'].items():
                    key = 'module.' + key
                    D_state_dict[key] = value
                self.D.load_state_dict(D_state_dict)
            else:
                self.G.load_state_dict(checkpoint['G_net'])
                self.D.load_state_dict(checkpoint['D_net'])
        else:
            # save on GPU, load on CPU
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            self.G.load_state_dict(checkpoint['G_net'])
            self.D.load_state_dict(checkpoint['D_net'])

        print("=========== loaded trained models (path: {})! ===========".format(checkpoint_path))


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_path)


    def identity_loss(self, idt_loss_type):
        if idt_loss_type == 'l1':
            criterion = nn.L1Loss()
            return criterion
        elif idt_loss_type == 'smoothl1':
            criterion = nn.SmoothL1Loss()
            return criterion
        elif idt_loss_type == 'l2':
            criterion = nn.MSELoss()
            return criterion
        else:
            raise NotImplementedError("=== Identity loss type [{}] is not implemented. ===".format(self.args.idt_loss_type))

