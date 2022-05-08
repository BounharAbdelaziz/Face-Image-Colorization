from model.base_nets.base_model_gan import BaseModelGAN
import torch
import torchvision

from model.loss import L2Loss, L1Loss, PerceptualLoss
import utils.helpers as helper
from tqdm import tqdm
import os

class ColorizationGAN(BaseModelGAN):

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#  

  def __init__(self, generator, discriminator, hyperparams, options) -> None:
      
      super(ColorizationGAN, self).__init__(generator, discriminator, options, hyperparams)

      # Loss functions
      if self.options.lambda_MSE:
          self.loss_MSE = L2Loss(device=self.hyperparams.device).to(self.hyperparams.device)
          self.loss_names.append("loss_MSE")

      if self.options.lambda_L1:
          self.loss_L1 = L1Loss(device=self.hyperparams.device).to(self.hyperparams.device)    
          self.loss_names.append("loss_L1")

      if self.options.lambda_PCP:
          self.loss_PCP = PerceptualLoss(device=self.hyperparams.device).to(self.hyperparams.device)
          self.loss_names.append("loss_PCP")

      print(f'[INFO] Using losses : {self.loss_names}')

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#
    
  def backward_G(self, disc_fake, fake, real):

    losses = {}
    loss_G_total = 0

    # Generator adversarial loss
    loss_G = self.options.lambda_G * self.loss_G(disc_fake)
    losses["loss_G"] = loss_G

    loss_G_total = loss_G_total + loss_G

    # Perceptual loss
    if self.options.lambda_PCP:
      loss_PCP = self.options.lambda_PCP * self.loss_PCP(fake, real)
      loss_G_total = loss_G_total + loss_PCP
      losses["loss_PCP"] = loss_PCP

    # MSE loss
    if self.options.lambda_MSE:
      loss_MSE =  self.options.lambda_MSE * self.loss_MSE(fake, real)
      loss_G_total = loss_G_total + loss_MSE
      losses["loss_MSE"] = loss_MSE

    # L1 loss
    if self.options.lambda_L1:
      loss_L1 =  self.options.lambda_L1 * self.loss_L1(fake, real)
      loss_G_total = loss_G_total + loss_L1
      losses["loss_L1"] = loss_L1

    losses["loss_G_total"] = loss_G_total
   
    with torch.autograd.set_detect_anomaly(self.options.detect_anomaly) : # set to True only during debug
      loss_G_total.backward()

    return losses

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def backward_D(self, disc_real, disc_fake):
    
    # Discriminator adversarial loss
    loss_D = self.loss_D(disc_real, disc_fake)
    loss_D = self.options.lambda_D * loss_D

    with torch.autograd.set_detect_anomaly(self.options.detect_anomaly) : # set to True only during debug
      loss_D.backward(retain_graph=True)

    return loss_D

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
      
      for batch_idx, data in enumerate(dataloader) : 

        image_rgb, image_gray = data

        with torch.autograd.set_detect_anomaly(self.options.detect_anomaly) :

          # Put data on available device (GPU or CPU)
          image_rgb = image_rgb.float().to(self.hyperparams.device)
          image_gray = image_gray.float().to(self.hyperparams.device)

          # we generate an image according to the age
          fake_image_rgb = self.generator(image_gray)

          # prediction of the discriminator on real and fake images in the batch
          disc_real = self.discriminator(image_rgb)
          # detach from the computational graph to re-use the output of the Generator
          disc_fake = self.discriminator(fake_image_rgb)

          # Optimizing the Discriminator
          self.opt_disc.zero_grad()
          loss_D = self.backward_D(disc_real, disc_fake)
          self.opt_disc.step()

          # Optimizing the Generator
          disc_fake = self.discriminator(fake_image_rgb)
          self.opt_gen.zero_grad()
          losses = self.backward_G(disc_fake, fake_image_rgb, image_rgb)       
          self.opt_gen.step()
          
          step = step + 1
          # Logging advances

          if batch_idx % self.hyperparams.show_advance == 0 and batch_idx!=0:

            # show advance
            with torch.no_grad():
              print(f"[INFO] Learnable alpha : {self.generator.alpha}")
              print(f"[INFO] Learnable beta : {self.generator.beta}")

              
              fake_rgb = fake_image_rgb[0][:3, :, :].reshape(1, 3, h, w)
              real_rgb = image_rgb[0][:3, :, :].reshape(1, 3, h, w)
              real_gray = image_gray[0][:1, :, :].reshape(1, 1, h, w)

              img_fake = torchvision.utils.make_grid(fake_rgb, normalize=True)
              img_real = torchvision.utils.make_grid(real_rgb, normalize=True)
              img_real_gray = torchvision.utils.make_grid(real_gray, normalize=True)

              losses["loss_D"] = loss_D
              
              # lr schedulers
              losses["lr_gen"] = helper.get_last_lr(self.opt_gen)
              losses["lr_disc"] = helper.get_last_lr(self.opt_disc)

              helper.write_logs_tb(self.tb_writer_loss, self.tb_writer_fake, self.tb_writer_real, img_fake, img_real_gray, img_real, losses, step, epoch, self.hyperparams, with_print_logs=False, experiment=self.options.experiment_name)


        if batch_idx % self.hyperparams.save_weights == 0 and batch_idx!=0 :

          # Saving weights
          print("[INFO] Saving weights...")
          torch.save(self.discriminator.state_dict(), os.path.join(self.PATH_CKPT,"D_it_"+str(step)+".pth"))
          torch.save(self.generator.state_dict(), os.path.join(self.PATH_CKPT,"G_it_"+str(step)+".pth"))

      # Learning rate scheduler
      self.scheduler_gen.step(self.scheduler_gen.last_epoch+1)
      self.scheduler_disc.step(self.scheduler_disc.last_epoch+1)

      if self.options.warmup_period:

        self.warmup_scheduler_gen.dampen()
        self.warmup_scheduler_disc.dampen()

    print("[INFO] Saving weights last step...")
    torch.save(self.discriminator.state_dict(), os.path.join(self.PATH_CKPT,"D_last_it_"+str(step)+".pth"))
    torch.save(self.generator.state_dict(), os.path.join(self.PATH_CKPT,"G_last_it_"+str(step)+".pth"))
    print(f'Latest networks saved in : {self.PATH_CKPT}')



    