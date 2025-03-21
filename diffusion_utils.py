import torch
import numpy as np

class EDMPrecond(torch.nn.Module):
    """ Original Class:: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L519
    Adapted from Chuck White (CHW) who adapted from Randy Chase (RJC).
    
    This is a wrapper for your pytorch model. Its purpose is to apply the preconditioning that is talked about in Karras et al. (2022)'s EDM paper.
    That is, apply clever scaling to speed up the diffusion process.
    """
    def __init__(self,
        img_resolution,                     # Image resolution in (H, W)
        img_channels,                       # Number of channels to be diffused
        model,                              # PyTorch denoising model
        use_fp16        = True,             # Execute the underlying model at FP16 precision
        sigma_data      = 0.5,              # Expected standard deviation of the training data; this was default from original class
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x, sigma, force_fp32=False, **model_kwargs):
        
        """ 
        Calls denoising network after applying preconditioning scaling from Karras et al. (2022)'s EDM paper 
        (see Appendix B.6).
        
        :param x: PyTorch tensor - Noised image channels concatenated with conditioning image channels
                  x[:, 0:self.img_channels, ...] should be noised images. The rest are conditions.
                  x is of shape batch, channel, nx, ny
        :param sigma: float or tensor of foats - Position in noise space. sigma == 0 is no noise, sigma >> 0 is all noise
                      If a tensor, then each sigma value corresponds to a particular element in a batch.
        :kwarg force_fp32: bool - Holdover from original class to force the network's precision
        """

        #start in float32 b/c preconditioning scaling is all done with that precision
        x = x.to(torch.float32)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        #reshape sigma to match the B, C, H, W dimensionality of x
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        #get scaling weights from EDM 
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        #split the noisy image from conditions so input scaling can be applied only to the noisy image
        x_noisy = torch.clone(x[:,0:self.img_channels,...])
        x_condition = torch.clone(x[:,self.img_channels:,...])
        
        #concatenate back with the scaling applied to the noisy image 
        model_input_images = torch.cat([x_noisy*c_in, x_condition], dim=1)
        
        #predict noise with the denoising model given scaled inputs and sigma
        F_x = self.model((model_input_images).to(dtype), c_noise.flatten())
        
        #apply remaining scaling from EDM to get image with less noise
        D_x = c_skip * x_noisy + c_out * F_x.to(torch.float32)
        
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class EDMLoss:
    
    """Original Class:: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py
    Adapted from CHW who adapted from RJC.

    Loss function for training the denoiser model. Note that the noise level sampling is encapsulated here.
    """
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        """
        P refers to the (log-normal) distribution from which random sigma levels are chosen.
        Per CHW: Corrdiff and/or gencast: P_mean=0, P_std=1.2, sigma_data=1
                 EDM default: P_mean=-1.2, P_std=1.2, sigma_data=0.5
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data #I believe this should just be 1 if you've standardized your input data... open to correction

    def __call__(self, wrapped_denoiser, clean_images, condition_images):
        
        """ 
        For each sample, chooses a random noise level and asks the wrapped denoiser to fully denoise a clean image
        from that noise level. The loss is then a weighted squared error between the denoised and clean images. 
        The weight grows exponentially as sigma approaches 0 and approaches (1/sigma_data ** 2) as sigma approaches
        infinity.
        
        :param wrapped_denoiser: EDMPrecond wrapping a PyTorch model - NN for predicting noise to remove from an image
        :param clean_images: PyTorch tensor of dimension [batch,channel,nx,ny] containing soon-to-be-noised images
        :param condition_images: PyTorch tensor of dimension [batch,channel,nx,ny] containing conditional images
        
        """
        
        ##get random noise levels (sigmas) for each sample in the batch
        rnd_normal = torch.randn([clean_images.shape[0], 1, 1, 1], device=clean_images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp() #transform to samples from a log-normal distribution
        
        #get the loss weight for those sigmas 
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #add noise to the batch of clean images 
        n = torch.randn_like(clean_images) * sigma
        noisy_images = torch.clone(clean_images + n)
        
        #cat the noisy images and conditioning images on the channel dimension for the wrapped model call 
        model_input_images = torch.cat([noisy_images, condition_images], dim=1)
        
        #call the EDMPrecond-wrapped model 
        denoised_images = wrapped_denoiser(model_input_images, sigma) # net is the preconditioned model
        
        #calc the weighted loss at each pixel
        loss = weight * ((denoised_images - clean_images) ** 2) #note this is still (batch, channel, nx, ny)!
        
        return loss


class StackedRandomGenerator:
    """
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a batch. Main use case is generating "latent" random
    noise for sampling a batch of images.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(f"Expected first dimension of size {len(self.generators)}, got {size[0]}")
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(f"Expected first dimension of size {len(self.generators)}, got {size[0]}")
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs)for gen in self.generators])
    
def edm_sampler(
    wrapped_denoiser, latents, condition_images, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """ 
    Adapted from: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py
    Subsequently adapted from CHW who adapted from RJC.

    Applies Heun's method (see Appendix D.2 of Karras 2022) to iteratively remove noise from an image.
    
    :param wrapped_denoiser: EDMPrecond wrapping a PyTorch model - NN for predicting noise to remove from an image
    :param latents: PyTorch tensor - Random noise from StackedRandomGenerator of dimension [batch,channel,nx,ny]
    :param condition_images:PyTorch tensor of dimension [batch,channel,nx,ny] containing conditional images
    
    [Timestep Discretization Parameters]
    :kwarg num_steps: int - Number of steps to take to denoise from latent noise
    :kwarg sigma_min: float - Minimum sigma value to step towards before sigma == 0
    :kwarg sigma_max: float - Sigma value to begin iteration from
    :kwarg rho: float - Controls how much steps near sigma_min are shortened at the expense of longer steps near 
                        sigma_max. Per Karras 2022, rho between 5 and 10 performs optimally for sampling.
    
    [SDE Solver Parameters]
    :kwarg randn_like: torch random number generator - Used to add noise if S_churn > 0 (i.e. taking the SDE approach)
    :kwarg S_churn: float - Controls magnitude of noise to add
    :kwarg S_min: float - Minimum sigma level at which additional noise will be added
    :kwarg S_max: float - Maximum sigma level at which additional noise will be added
    :kwarg S_noise: float -Standard deviation of normal distribution sampled for noise. Usually 1, but can be slightly larger.    
    """
    
    batch_size = condition_images.shape[0]

    # Time step discretization. 
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([wrapped_denoiser.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # sets final sigma to 0.
    
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily if doing the SDE route
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = wrapped_denoiser.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        #Concatenate conditional images
        model_input_images = torch.cat([x_hat, condition_images], dim=1)
        
        # Euler step.
        with torch.no_grad():
            denoised = wrapped_denoiser(model_input_images, t_hat.unsqueeze(0).expand(batch_size, 1, 1, 1)).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            model_input_images = torch.cat([x_next, condition_images], dim=1)
            with torch.no_grad():
                denoised = wrapped_denoiser(model_input_images, t_next.unsqueeze(0).expand(batch_size, 1, 1, 1)).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next