import torch
import torch.nn as nn

from model.block import Conv2DLayer


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, features=[64, 128, 256, 512], norm_type="in2d", **kwargs):
        super(Discriminator, self).__init__()

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Middle layers
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Conv2DLayer(  in_channels, out_features=feature, kernel_size=3, bias=True, norm_type=norm_type, norm_before=True, activation='lk_relu', 
                                        inplace=True, padding_mode="reflect", scale='down', **kwargs))         
                                        
            in_channels = feature

        # Output layer
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect", **kwargs))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.sigmoid(self.model(x))
        
        return x