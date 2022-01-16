import torch.nn as nn
import torchvision.models as models



# -----------------------------------------------------------------------------#
#                     Perceptual Feature extraction Module                     #
# -----------------------------------------------------------------------------#

class VGG_19(nn.Module):

    def __init__(self, extraction_layer=''):
        super(VGG_19, self).__init__()
        self.vgg_net = models.vgg19(pretrained=True)
        self.extraction_layer = extraction_layer

    def forward(self, x):
        return x


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
