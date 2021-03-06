import torch.nn as nn
from model.networks import VGG_19
from utils.helpers import *


class L2Loss(nn.Module):
    def __init__(self, device='cuda'):
        super(L2Loss, self).__init__()
        self.eps = 1e-8
        self.device = device
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
         
        return self.criterion(x,y) + self.eps

class L1Loss(nn.Module):
    def __init__(self, device='cuda'):
        super(L1Loss, self).__init__()
        self.eps = 1e-8
        self.device = device
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
         
        return self.criterion(x,y) + self.eps       

class GenLoss(nn.Module):
  
    def __init__(self, gan_type='vanilla' ):
        
        super(GenLoss, self).__init__()
        self.gan_type = gan_type

        # vanilla GAN, min_D max_G log(D(x)) + log(1 - D(G(z)))
        if self.gan_type == 'vanilla' :
            self.criterion = nn.BCELoss()

        # least square error
        elif self.gan_type == 'lsgan' :
            self.criterion = nn.MSELoss()
        
        # Wasserstein GAN tackles the problem of Mode Collapse and Vanishing Gradient. 
        elif self.gan_type == 'wgan':
            self.criterion = None

        else:
            raise NotImplementedError('[INFO] The GAN type %s is not implemented !' % self.gan_type)

        self.eps = 1e-8

    def forward(self, disc_pred):

        ##########################################
        #####         Generator Loss          ####
        ##########################################
        
        if self.gan_type in ['vanilla', 'lsgan'] :

            # we want to min log(1 - D(G(z))) which is equivalent to max log(D(G(z))) and it's better in the begining of the training (better gradients).
            loss_G = self.criterion(disc_pred, torch.ones_like(disc_pred))

        else:
            raise NotImplementedError('[INFO] The GAN type %s is not implemented !' % self.gan_type)
        return loss_G + self.eps


class DiscLoss(nn.Module):
  
    def __init__(self, gan_type='vanilla' ):
        
        super(DiscLoss, self).__init__()
        self.gan_type = gan_type

        # vanilla GAN, min_D max_G log(D(x)) + log(1 - D(G(z)))
        if self.gan_type == 'vanilla' :
            self.criterion = nn.BCELoss()

        # least square error
        elif self.gan_type == 'lsgan' :
            self.criterion = nn.MSELoss()
        
        # Wasserstein GAN tackles the problem of Mode Collapse and Vanishing Gradient. 
        elif self.gan_type == 'wgan':
            self.criterion = None

        else:
            raise NotImplementedError('[INFO] The GAN type %s is not implemented !' % self.gan_type)

        self.eps = 1e-8
        
    def forward(self, disc_real, disc_fake):

        ##########################################
        #####       Discriminator Loss        ####
        ##########################################

        if self.gan_type in ['vanilla', 'lsgan'] :

            # the loss on the real image in the batch
            loss_D_real = self.criterion(disc_real, torch.ones_like(disc_real))
            # the loss on the fake image in the batch
            loss_D_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
            # total discriminator loss          
            loss_D = (loss_D_real + loss_D_fake) / 2

        return loss_D + self.eps


class PerceptualLoss(nn.Module):

    """ The perceptual loss combines the MSE (pixel-domain) loss and extracted features by the VGG-19 network """
    def __init__(self, feature_layers=[0, 3, 8, 17, 26, 35], device="cuda:0"):
        super(PerceptualLoss, self).__init__()
        # Feature-domain Loss
        self.vgg_net = VGG_19().to(device)
        self.feature_layers = feature_layers
        self.build_features_extractor()
        self.criterion_features = nn.MSELoss()
        self.eps = 1e-8
        # Pixel-domain Loss
        self.mse = nn.MSELoss()

        for param in self.vgg_net.parameters():
            param.requires_grad = False
               
    def build_features_extractor(self):
        # print(self.vgg_net)
        pretrained_features = self.vgg_net.vgg_net.features

        self.features_extractor = []
        for i in range(len(self.feature_layers)-1): # -1 to prevent from index out of range (cf. next for loop; self.feature_layers[i+1])

            # we stack the layers and form a small sequential network that will extract features at a given scale
            scale_network = torch.nn.Sequential()
            for j in range(self.feature_layers[i], self.feature_layers[i+1]):
                scale_network.add_module(str(j), pretrained_features[j])
            self.features_extractor.append(scale_network)

        self.features_extractor = torch.nn.ModuleList(self.features_extractor)

    def get_features(self, x):

        features = []
        for f in self.features_extractor :
            x = f(x)
            features.append(x)

        return features

    def preprocess(self, x):

        # ImageNet mean and std
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(x.device)
        mean = mean.view(1, 3, 1, 1)

        std = torch.Tensor([0.229, 0.224, 0.225]).to(x.device)
        std = std.view(1, 3, 1, 1)

        # center the image variable to a zero-mean and 1-std
        x = (x-mean)/std

        if x.shape[3] <224 :
            x = torch.nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        
        return x

    def forward(self, x, y):

        x = self.preprocess(x)
        y = self.preprocess(y)

        x_features = self.get_features(x)
        y_features = self.get_features(y)
        
        pcp_loss = 0
        for xf, yf in zip(x_features, y_features):
            pcp_loss = pcp_loss + self.criterion_features(xf, yf)

        L2_loss = self.mse(x, y)

        # total loss
        total_loss = pcp_loss + L2_loss + self.eps

        return total_loss