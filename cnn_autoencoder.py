import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from timm.models.layers import DropPath

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvBlock(nn.Module):
    def __init__(self, channels_in = 1, channels_out = 1, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, padding=kernel_size//2)
        self.norm = LayerNorm(channels_out, eps=1e-6)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Convolves x with the block's kernel, normalizes, then passes through activation functon
        """

        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C) to allow norm to work
        x = self.norm(x) #TODO: Try BatchNorm
        x = self.act(x) #TODO: Use LeakyReLU with slope param [.01, .2]
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x


class ConvAutoEncoder(nn.Module):
    def __init__(self, n_input_channels=1, n_output_channels=1, n_conv_filters=[16, 16], n_layers_per_block=1):
        super().__init__()

        #setting hyperparameter class attrs
        self.n_encoder_layers = len(n_conv_filters) #the last index is number of bottleneck filters
        
        #helper layers
        self.pool = nn.AvgPool2d((2, 2), stride=2)
        self.final = nn.Conv2d(n_conv_filters[0], n_output_channels, kernel_size=1)

        #Encoder layers
        #Input is passed through n_layers_per_block convolutional blocks before being pooled (reducing dimension by //2)
        #The number of times this is repeated is modulated via the length of n_cov_filters which controls the number 
        #of convolutional filters at each pass
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(*[
                ConvBlock(
                    channels_in=n_input_channels if i == 0 and j == 0 else n_conv_filters[i - (1 if j == 0 else 0)], 
                    channels_out = n_conv_filters[i], 
                    kernel_size=3
                )
                for j in range(n_layers_per_block)
            ])
            for i in range(self.n_encoder_layers)
        ])
        
        # Decoder layers
        #Input is first passed through a transposed convolution (increasing the dimension *2)
        #This upscaled output is then passed through a set of convolutional filters IF n_layers_per_block >1
        self.decoder_layers = nn.ModuleList()
        for i in range(self.n_encoder_layers - 1, -1, -1):
            in_channels = n_conv_filters[i]
            out_channels = n_conv_filters[i - 1] if i > 0 else n_conv_filters[0]  # Keep first layer wide
            
            # First decoder layer maintains channels
            if i == self.n_encoder_layers - 1:
                out_channels = in_channels  # Keep channels unchanged for first upsampling

            upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

            conv_blocks = [
                ConvBlock(channels_in=out_channels, channels_out=out_channels, kernel_size=3)
                for _ in range(n_layers_per_block - 1)
            ]

            self.decoder_layers.append(nn.Sequential(upsample, *conv_blocks))

        
    def forward(self, inputs):

        #logic for concatenating multiple channels
        x = torch.cat((inputs,), dim=1) if isinstance(inputs, torch.Tensor) else torch.cat(inputs, dim=1) #edge case for just a single channel

        #encode to latent space
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
            x = self.pool(x)
 
        #decode from latent space
        for dec_layer in self.decoder_layers:
            x = dec_layer(x)

        #final 1x1 convolution
        output = self.final(x)
        return output