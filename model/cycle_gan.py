from model.base_nets.base_model import BaseModel
import torch
import torch.nn as nn
import torchvision

from model.generator import Generator
from model.discriminator import Discriminator

import utils.helpers as helper
from tqdm import tqdm
import os
from model.optimization import get_optimizer, get_scheduler, get_lr_warmup

from torch.utils.tensorboard import SummaryWriter

class CycleGAN(BaseModel):

    """
        This class implements the CycleGAN model, for learning image-to-image translation without paired data.

        * CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    def __init__(self, hyperparams, options) -> None:
        super(CycleGAN, self).__init__()

        self.hyperparams = hyperparams
        self.options = options
        self.experiment = options.experiment_name

        self.loss_names = []
        self.loss_names.append("loss_G")
        self.loss_names.append("loss_D")

        self.eps = 1e-8

        # Gradient scaler
        self.g1_scaler = torch.cuda.amp.GradScaler()
        self.g2_scaler = torch.cuda.amp.GradScaler()
        self.d1_scaler = torch.cuda.amp.GradScaler()
        self.d2_scaler = torch.cuda.amp.GradScaler()

        # Tensorboard
        self.tb_writer_fake = SummaryWriter(f"logs/{self.experiment}/fake_{self.experiment}")
        self.tb_writer_real = SummaryWriter(f"logs/{self.experiment}/real_{self.experiment}")
        self.tb_writer_loss = SummaryWriter(f"logs/{self.experiment}/loss_train_{self.experiment}")

        # Generators
        # Let domain 1 be the RGB domain and domain 2 be the Gray domain
        # Generator 1 generates RGB images
        # Generator 2 generates Gray images

        self.gen_1 = Generator(in_channels=1, out_channels=3, n_res_layers=9, norm_type=options.norm_type).to(self.hyperparams.device)
        self.gen_2 = Generator(in_channels=3, out_channels=1, n_res_layers=9, norm_type=options.norm_type).to(self.hyperparams.device)

        # Discriminators
        # Discriminator 1 discriminates RGB images
        # Discriminator 2 discriminates Gray images

        self.disc_1 = Discriminator(in_channels=3, norm_type=options.norm_type).to(self.hyperparams.device)
        self.disc_2 = Discriminator(in_channels=1, norm_type=options.norm_type).to(self.hyperparams.device)

        helper.setup_network(self.gen_1, self.hyperparams, type="generator")
        helper.setup_network(self.gen_2, self.hyperparams, type="generator")
        helper.setup_network(self.disc_1, self.hyperparams, type="discriminator")
        helper.setup_network(self.disc_2, self.hyperparams, type="discriminator")

        # Optimizers
        self.opt_disc_1 = get_optimizer(self.disc_1, self.options)
        self.opt_disc_2 = get_optimizer(self.disc_2, self.options)
        self.opt_gen_1 = get_optimizer(self.gen_1, self.options)
        self.opt_gen_2 = get_optimizer(self.gen_2, self.options)

        # Learning Rate Scheduler
        self.scheduler_g1 = get_scheduler(self.opt_gen_1, options)
        self.scheduler_g2 = get_scheduler(self.opt_gen_2, options)
        self.scheduler_d1 = get_scheduler(self.opt_disc_1, options)
        self.scheduler_d2 = get_scheduler(self.opt_disc_2, options)

        # Learning Rate Warmup stage
        if options.warmup_period:
            self.warmup_scheduler_g1 = get_lr_warmup(self.opt_gen_1, options)
            self.warmup_scheduler_g2 = get_lr_warmup(self.opt_gen_2, options)
            self.warmup_scheduler_d1 = get_lr_warmup(self.opt_disc_1, options)
            self.warmup_scheduler_d2 = get_lr_warmup(self.opt_disc_2, options)

        # Loss functions
        self.loss_cycle = nn.L1Loss().to(self.hyperparams.device)
        self.loss_L2 = nn.MSELoss().to(self.hyperparams.device)
        print(f'[INFO] Using losses : {self.loss_names}')

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, dataloader):

        h = self.options.img_size
        w = self.options.img_size

        self.PATH_CKPT = self.options.checkpoints_dir + self.options.experiment_name+"/"
        
        print("[INFO] Started training using device ",self.hyperparams.device,"...")      

        step = 0
        for epoch in tqdm(range(self.hyperparams.n_epochs)):
            print("epoch = ",epoch," --------------------------------------------------------\n")

            for batch_idx, data in enumerate(dataloader):
                
                image_rgb, image_gray = data
                image_rgb = image_rgb.to(self.hyperparams.device)
                image_gray = image_gray.to(self.hyperparams.device)

                # Let domain 1 be the RGB domain and domain 2 be the Gray domain
                # Generator 1 generates RGB images
                # Generator 2 generates Gray images
                # Discriminator 1 discriminates RGB images
                # Discriminator 2 discriminates Gray images

                with torch.autograd.set_detect_anomaly(self.options.detect_anomaly):
                    
                    # Train Discriminators 1 and 2
                    with torch.cuda.amp.autocast():

                        # From domain 1 to domain 2 (RGB -> GRAY)
                        fake_img_rgb = self.gen_1(image_gray) # takes an image in domain 2 and generates one in domain 1
                        D_1_real = self.disc_1(image_rgb) # discriminates real/fake images in domain 1
                        D_1_fake = self.disc_1(fake_img_rgb.detach())
                        # Least-square GAN by default
                        D_1_real_loss = self.loss_L2(D_1_real, torch.ones_like(D_1_real))
                        D_1_fake_loss = self.loss_L2(D_1_fake, torch.zeros_like(D_1_fake))
                        D_1_loss = (D_1_real_loss + D_1_fake_loss) * self.options.lambda_D + self.eps

                        # From domain 2 to domain 1 (GRAY -> RGB)
                        fake_img_gray = self.gen_2(image_rgb) # takes an image in domain 1 and generates one in domain 2
                        D_2_real = self.disc_2(image_gray) # discriminates real/fake images in domain 2
                        D_2_fake = self.disc_2(fake_img_gray.detach())
                        D_2_real_loss = self.loss_L2(D_2_real, torch.ones_like(D_2_real))
                        D_2_fake_loss = self.loss_L2(D_2_fake, torch.zeros_like(D_2_fake))
                        D_2_loss = (D_2_real_loss + D_2_fake_loss) * self.options.lambda_D + self.eps


                    self.opt_disc_1.zero_grad()
                    self.d1_scaler.scale(D_1_loss).backward()
                    self.d1_scaler.step(self.opt_disc_1)
                    self.d1_scaler.update()

                    self.opt_disc_2.zero_grad()
                    self.d2_scaler.scale(D_2_loss).backward()
                    self.d2_scaler.step(self.opt_disc_2)
                    self.d2_scaler.update()

                    # Train Generators 1 and 2
                    with torch.cuda.amp.autocast():

                        # adversarial loss for both generators
                        D_1_fake = self.disc_1(fake_img_rgb) # discriminates real/fake images in domain 2 (GRAY)
                        D_2_fake = self.disc_2(fake_img_gray) # discriminates real/fake images in domain 1 (RGB)

                        # Least-square GAN by default
                        loss_G_1 = self.loss_L2(D_1_fake, torch.ones_like(D_1_fake))
                        loss_G_2 = self.loss_L2(D_2_fake, torch.ones_like(D_2_fake))

                        # cycle loss
                        cycle_rgb = self.gen_1(fake_img_gray) # G1 takes a fake image in domain 2 (GRAY) and generates back the original input image in domain 1 (RGB)
                        cycle_gray = self.gen_2(fake_img_rgb) # G2 takes a fake image in domain 1 (RGB) and generates back the original input image in domain 2 (GRAY)
                        cycle_1_loss = self.loss_cycle(image_rgb, cycle_rgb) + self.eps
                        cycle_2_loss = self.loss_cycle(image_gray, cycle_gray) + self.eps

                        # Total loss
                        loss_G1_total = loss_G_1 * self.options.lambda_G + cycle_1_loss * self.options.lambda_cycle
                        loss_G2_total = loss_G_2 * self.options.lambda_G + cycle_2_loss * self.options.lambda_cycle
                        
                    self.opt_gen_1.zero_grad()
                    self.g1_scaler.scale(loss_G1_total).backward(retain_graph=True)
                    self.g1_scaler.step(self.opt_gen_1)
                    self.g1_scaler.update()

                    self.opt_gen_2.zero_grad()
                    self.g2_scaler.scale(loss_G2_total).backward()
                    self.g2_scaler.step(self.opt_gen_2)
                    self.g2_scaler.update()
                

                # Logging advances
                if batch_idx % self.hyperparams.show_advance == 0 and batch_idx!=0:

                    # show advance in tensorboard
                    with torch.no_grad():
                    
                        fake_img_rgb = fake_img_rgb[0][:3, :, :].reshape(1, 3, h, w)
                        fake_img_gray = fake_img_gray[0][:3, :, :].reshape(1, 1, h, w)

                        image_rgb = image_rgb[0][:3, :, :].reshape(1, 3, h, w)
                        image_gray = image_gray[0][:3, :, :].reshape(1, 1, h, w)

                        cycle_rgb = cycle_rgb[0][:3, :, :].reshape(1, 3, h, w)
                        cycle_gray = cycle_gray[0][:3, :, :].reshape(1, 1, h, w)

                        img_fake_rgb = torchvision.utils.make_grid(fake_img_rgb, normalize=True)
                        img_fake_gray = torchvision.utils.make_grid(fake_img_gray, normalize=True)

                        img_cycle_rgb = torchvision.utils.make_grid(cycle_rgb, normalize=True)
                        img_cycle_gray = torchvision.utils.make_grid(cycle_gray, normalize=True)

                        img_real_rgb = torchvision.utils.make_grid(image_rgb, normalize=True)
                        img_real_gray = torchvision.utils.make_grid(image_gray, normalize=True)
                        
                        images_fakes = [img_fake_rgb, img_fake_gray, img_cycle_rgb, img_cycle_gray]
                        images_real = [img_real_rgb, img_real_gray]

                        losses = {}
                        # Computed losses
                        losses["loss_D1"] = D_1_loss
                        losses["loss_D2"] = D_2_loss
                        losses["loss_G1"] = loss_G_1
                        losses["cycle_1_loss"] = cycle_1_loss
                        losses["cycle_2_loss"] = cycle_2_loss
                        losses["loss_G2"] = loss_G_2
                        losses["loss_G1_total"] = loss_G1_total
                        losses["loss_G2_total"] = loss_G2_total
                        
                        # lr schedulers
                        losses["lr_g1"] = helper.get_last_lr(self.opt_gen_1)
                        losses["lr_d1"] = helper.get_last_lr(self.opt_disc_1)
                        losses["lr_g2"] = helper.get_last_lr(self.opt_gen_2)
                        losses["lr_d2"] = helper.get_last_lr(self.opt_disc_2)
                        
                        helper.write_logs_tb_cyclegan(self.tb_writer_loss, self.tb_writer_fake, self.tb_writer_real, images_fakes, images_real, losses, step, epoch, self.hyperparams, with_print_logs=False, experiment=self.experiment)

                        step = step + batch_idx


                if batch_idx % self.hyperparams.save_weights == 0 and batch_idx!=0 :

                    # Saving weights
                    print("[INFO] Saving weights...")
                    torch.save(self.disc_1.state_dict(), os.path.join(self.PATH_CKPT,"D1_it_"+str(step)+".pth"))
                    torch.save(self.disc_2.state_dict(), os.path.join(self.PATH_CKPT,"D2_it_"+str(step)+".pth"))
                    torch.save(self.gen_1.state_dict(), os.path.join(self.PATH_CKPT,"G1_it_"+str(step)+".pth"))
                    torch.save(self.gen_2.state_dict(), os.path.join(self.PATH_CKPT,"G2_it_"+str(step)+".pth"))

            # Learning rate scheduler
            self.scheduler_g1.step(self.scheduler_g1.last_epoch+1)
            self.scheduler_g2.step(self.scheduler_g2.last_epoch+1)
            self.scheduler_d1.step(self.scheduler_d1.last_epoch+1)
            self.scheduler_d2.step(self.scheduler_d2.last_epoch+1)

            if self.options.warmup_period:
                self.warmup_scheduler_g1.dampen()
                self.warmup_scheduler_g2.dampen()
                self.warmup_scheduler_d1.dampen()
                self.warmup_scheduler_d2.dampen()
            

        print("[INFO] Saving weights last step...")
        torch.save(self.disc_1.state_dict(), os.path.join(self.PATH_CKPT,"D1_it_"+str(step)+".pth"))
        torch.save(self.disc_2.state_dict(), os.path.join(self.PATH_CKPT,"D2_it_"+str(step)+".pth"))
        torch.save(self.gen_1.state_dict(), os.path.join(self.PATH_CKPT,"G1_it_"+str(step)+".pth"))
        torch.save(self.gen_2.state_dict(), os.path.join(self.PATH_CKPT,"G2_it_"+str(step)+".pth"))

        print(f'Latest networks saved in : {self.PATH_CKPT}')