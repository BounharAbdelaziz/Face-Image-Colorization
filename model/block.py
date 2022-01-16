import torch.nn as nn

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
     
class ActivationLayer(nn.Module):

    def __init__(self, activation='lk_relu', alpha_relu=0.15, inplace=False):
        super().__init__()

        if activation =='lk_relu':
            self.activation = nn.LeakyReLU(alpha_relu)

        elif activation =='relu':
            self.activation = nn.ReLU(inplace)

        elif activation =='softmax':
            self.activation = nn.Softmax()

        elif activation =='sigmoid':
            self.activation = nn.Sigmoid()

        elif activation =='tanh':
            self.activation = nn.Tanh()

        elif activation =='selu':
            self.activation = nn.Selu()

        else :
            # Identity function
            self.activation = None

    def forward(self, x):

        if self.activation is None :
            # Identity function
            return x
        
        else :
            return self.activation(x)

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

class NormalizationLayer(nn.Module):

    def __init__(self, in_features, norm_type='bn1d'):
        super().__init__()

        if norm_type == 'bn2d' :
            self.norm = nn.BatchNorm2d(in_features)

        elif norm_type == 'in2d' :
            self.norm = nn.InstanceNorm2d(in_features)

        elif norm_type == 'bn1d' :
            self.norm = nn.BatchNorm1d(in_features)

        elif norm_type == 'in1d' :
            self.norm = nn.InstanceNorm1d(in_features)

        elif norm_type == 'none' :
            self.norm = lambda x : x * 1.0

        else:
            raise NotImplementedError('[INFO] The Normalization layer %s is not implemented !' % norm_type)

    def forward(self, x):
        out = self.norm(x)
        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

class Conv2DLayer(nn.Module):
    def __init__(   self, in_features, out_features, scale='none', norm_type='in2d', norm_before=True, activation='lk_relu', alpha_relu=0.15, 
                    interpolation_mode='bicubic', inplace=False, scale_factor=2, is_debug=False, **kwargs):
        
        super().__init__()
        
        # Sometimes, doing normalization before activation helps stabilizing the training
        self.norm_before = norm_before
        self.scale_factor = scale_factor
        self.is_debug = is_debug
        
        # upsampling or downsampling
        stride = 2 if scale == 'down' else 1

        if scale == 'up':
            self.scale_layer = lambda x : nn.functional.interpolate(x, scale_factor=scale_factor, mode=interpolation_mode)
        else :
            self.scale_layer = lambda x : x

        # Convolutional layer
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, stride=stride, **kwargs)

        # Activation layer
        self.activation = ActivationLayer(activation=activation, alpha_relu=alpha_relu, inplace=inplace)

        # Normalization layer
        self.norm = NormalizationLayer(in_features=out_features, norm_type=norm_type)

    def forward(self, x):
        
        # upsampling or downsampling 
        out = self.scale_layer(x)

        out = self.conv(out)

        if self.norm_before :
            out = self.norm(out)
            out = self.activation(out)

        else :
            out = self.activation(out)
            out = self.norm(out)

        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

class ConvResidualBlock(nn.Module):
    """ Residual blocks idea is to feed the output of one layer to another layer after a number of hops (generaly 2 to 3). Here we are using a hops of 2.
        It can be expressed in the form : F(x) + x, where x is the input and F modeling one or many layers.
        The benefits of using Residual Blocks is to overcome the vanishing gradients problem, and thus training very deep networks.
    """
    def __init__(   self, in_features, out_features, scale='none', norm_type='in2d', norm_before=True, activation='lk_relu', 
                    alpha_relu=0.15, interpolation_mode='nearest', use_act_second=True, scale_factor=2, is_debug=False,  **kwargs):
        super().__init__()

        self.is_debug = is_debug
        # Sometimes, doing normalization before activation helps stabilizing the training
        self.norm_before = norm_before

        # Computing the shortcut, we can use convolution or only identity
        if scale == 'none' and in_features == out_features :
            self.identity = lambda x : x
        else :
            self.identity = Conv2DLayer(in_features=in_features, out_features=out_features, scale=scale, norm_type=norm_type, norm_before=norm_before, 
                                        activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode, scale_factor=scale_factor, **kwargs)

        # defining the scaling, we don't want to do 2 up sampling or 2 downsampling in one block, therefore:
        if scale == 'none' :
            scales = ['none', 'none']

        if scale == 'up' :
            # start by upsampling
            scales = ['up', 'none']

        if scale == 'down' :
            # downsampling in the end
            scales = ['none', 'down']     


        # Convolutional layers that defines the Residual Block
        self.conv1 = Conv2DLayer(   in_features=in_features, out_features=out_features, scale=scales[0], norm_type=norm_type, norm_before=norm_before, 
                                    activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode, scale_factor=scale_factor, **kwargs)

        if 'stride' in kwargs:
            if kwargs['stride'] == 2 and scale == 'down' and scale != 'none' :
                kwargs['stride'] = 1

        if use_act_second:            
            self.conv2 = Conv2DLayer(in_features=out_features, out_features=out_features, scale=scales[1], norm_type=norm_type, 
                                    norm_before=norm_before, activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode, scale_factor=scale_factor, **kwargs)
        else:
            self.conv2 = Conv2DLayer(in_features=out_features, out_features=out_features, scale=scales[1], norm_type=norm_type, 
                                 norm_before=norm_before, activation='none', alpha_relu=alpha_relu, interpolation_mode=interpolation_mode, scale_factor=scale_factor, **kwargs)
    

    def forward(self, x):
        if self.is_debug:
            print("---------------------------------------")
            print(f'ConvResidualBlock input : {x.shape}')

        identity = self.identity(x)
        if self.is_debug:
            print(f'identity shape : {identity.shape}')

        out = self.conv1(x)
        if self.is_debug:
            print(f'conv1 out : {out.shape}')


        out = self.conv2(out)
        if self.is_debug:
            print(f'conv2 out : {out.shape}')


        out = out + identity
        if self.is_debug:
            print(f'out : {out.shape}')



        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#