import torch
import torch.nn as nn

from model.block import Conv2DLayer, ConvResidualBlock

class Generator(nn.Module):
    """ Generative network of CycleGAN"""
    
    def __init__(self, in_channels=1, out_channels=3, num_features=64, n_res_layers=9, norm_type="adain", activation='relu', **kwargs) -> None:
        super(Generator, self).__init__()
        self.initial = Conv2DLayer( in_channels, num_features, kernel_size=7, padding=3, padding_mode="reflect", scale='none', bias=True, norm_type=norm_type, norm_before=True, 
                                    activation=activation, alpha_relu=0.15, inplace=True, **kwargs)
        
        self.down_blocks = nn.ModuleList(
            [
                Conv2DLayer( num_features, num_features*2, kernel_size=3, padding=1, scale='down', bias=True, norm_type=norm_type, norm_before=True, 
                                    activation=activation, inplace=True, is_debug=True, **kwargs),

                Conv2DLayer( num_features*2, num_features*4, kernel_size=3, padding=1, scale='down', bias=True, norm_type=norm_type, norm_before=True, 
                                    activation=activation, inplace=True, is_debug=True, **kwargs),
            ]
        )

        self.res_blocks = nn.Sequential(
            *[  ConvResidualBlock( num_features*4, num_features*4, kernel_size=3, padding=1, scale='none', bias=True, norm_type=norm_type, norm_before=True, activation=activation, use_act_second=False,**kwargs)
                for _ in range(n_res_layers)
             ]
        )

        self.up_blocks = nn.ModuleList(
            [
                Conv2DLayer( num_features*4, num_features*2, kernel_size=3, padding=1, scale='up', scale_factor=2, bias=True, norm_type=norm_type, norm_before=True, 
                                    activation=activation, inplace=True, **kwargs),
                Conv2DLayer( num_features*2, num_features, kernel_size=3, padding=1, scale='up', scale_factor=2, bias=True, norm_type=norm_type, norm_before=True, 
                                    activation=activation, inplace=True, **kwargs),
            ]
        )

        self.last = nn.Conv2d(num_features, out_channels, kernel_size=7, padding=3, padding_mode="reflect")

    def forward(self, x):

        x = self.initial(x)

        for layer in self.down_blocks:
            x = layer(x)

        x = self.res_blocks(x)

        for layer in self.up_blocks:
            x = layer(x)
        
        return  torch.tanh(self.last(x))