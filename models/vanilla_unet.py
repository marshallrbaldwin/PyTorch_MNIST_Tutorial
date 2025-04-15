import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).

    From CHW.
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

#little helper class to enable concise conv block stacks
class CustomSequential(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input
        
class VanillaUNet(nn.Module):
    def __init__(self,
                 channels_in=1,
                 channels_out=1,
                 n_conv_filters=[16, 16, 16], # last element indicates # of bottleneck conv filters
                 n_layers_per_block=1, #number of convolutional blocks prior to up/downsampling
                 n_bottleneck_blocks=1 #number of convolutional blocks at smallest "bottleneck" resolution
                 ):
        
        super().__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.n_encoder_layers = len(n_conv_filters) - 1 #last one is for bottleneck

        self.pool = nn.AvgPool2d((2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(n_conv_filters[0], channels_out, kernel_size=(1, 1))
        
        #stack of encoder layers with multiple FilmedConvBlocks per layer
        self.encoder_layers = nn.ModuleList([
            CustomSequential(*[
                ConvBlock(channels_in=(self.channels_in if i == 0 and j == 0 else n_conv_filters[i - (1 if j == 0 else 0)]),
                              channels_out=n_conv_filters[i],
                              kernel_size=3)
                for j in range(n_layers_per_block)
            ])
            for i in range(self.n_encoder_layers)
        ])

        #stack of bottleneck Layers with multiple FilmedConvBlocks per layer
        self.bottleneck_layers = CustomSequential(*[
            ConvBlock(channels_in=n_conv_filters[self.n_encoder_layers - 1] if i == 0 else n_conv_filters[-1], 
                        channels_out=n_conv_filters[-1],
                        kernel_size=3)
            for i in range(n_bottleneck_blocks)
        ])

        #stack of decoder layers with multiple FilmedConvBlocks per layer
        self.decoder_layers = nn.ModuleList([
            CustomSequential(*[
                ConvBlock(channels_in=(n_conv_filters[i] + n_conv_filters[i - 1]) if j == 0 else n_conv_filters[i - 1],
                              channels_out=n_conv_filters[i - 1],
                              kernel_size=3)
                for j in range(n_layers_per_block)
            ])
            for i in range(self.n_encoder_layers, 0, -1)
        ])
        
        

    def forward(self, x):

        #loop through each encoder layer, saving output for decoding path
        enc_outputs = []
        for i,enc_layer in enumerate(self.encoder_layers):
            x = enc_layer(x)
            enc_outputs.append(x)
            x = self.pool(x)

        #do bottleneck convolutions
        x = self.bottleneck_layers(x)
 
        #loop through each decoder layer, concatenating encoder layer output as extra channels
        for i, dec_layer in enumerate(self.decoder_layers):
            x = self.upsample(x)
            x = torch.cat([x, enc_outputs[-(i + 1)]], dim=1)
            x = dec_layer(x)

        #pass through a final linear layer 
        x = self.final(x)
        
        return x

    @staticmethod
    def get_timestep_embedding(
        timesteps: torch.Tensor,
        embedding_dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 0,
        scale: float = 1,
        max_period: int = 10000
    ):
        """
        Original func:: https://github.com/huggingface/diffusers/blob/844221ae4e20a8939ee052f75874e284f75d4c5c/src/diffusers/models/embeddings.py#L27
        This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    
        Returns an [N x dim] Tensor of positional embeddings.
    
        :param timesteps: torch.Tensor - 1D tensor containing the float timesteps
        :param embedding_dim: int - Dimension of each timestep embedding vector
        :kwarg flip_sin_to_cos: bool - Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        :kwarg downscale_freq_shift: float - >0 emphasizes global connections; <0 emphasizes local connections
        :kwarg scale: float - Scales all of the embeddings
        :kwarg max_period: int - Controls max frequency of the embeddings (10_000 is from Attention is All You Need)
        """
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - downscale_freq_shift)
    
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
    
        # scale embeddings
        emb = scale * emb
    
        # concat sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
        # flip sine and cosine embeddings
        if flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    
        # zero pad
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb
    
    