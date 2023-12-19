""" Contribution: https://github.com/lucidrains/DALLE2-pytorch """

import torch
import torch.nn as nn
from torch.nn import functional as F
from dalle2.utils import extract, first, default

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule for diffusion models.

    This schedule is based on a cosine function and is proposed in the paper
    'Improved Denoising Diffusion Probabilistic Models'.
    
    Args:
        timesteps (int): The number of timesteps in the diffusion process.
        s (float): A hyperparameter for adjusting the schedule shape.

    Returns:
        torch.Tensor: The beta values for each timestep.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    Linear beta schedule for diffusion models.

    This schedule linearly interpolates beta values between a start and end value.
    
    Args:
        timesteps (int): The number of timesteps in the diffusion process.

    Returns:
        torch.Tensor: The beta values for each timestep.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def quadratic_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    Quadratic beta schedule for diffusion models.

    This schedule uses a quadratic function to interpolate beta values.
    
    Args:
        timesteps (int): The number of timesteps in the diffusion process.

    Returns:
        torch.Tensor: The beta values for each timestep.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float64) ** 2

def sigmoid_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    Sigmoid beta schedule for diffusion models.

    This schedule uses a sigmoid function to interpolate beta values.
    
    Args:
        timesteps (int): The number of timesteps in the diffusion process.

    Returns:
        torch.Tensor: The beta values for each timestep.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(-6, 6, timesteps, dtype=torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

class NoiseScheduler(nn.Module):
    """
    A noise scheduler for controlling the diffusion process in generative models.

    This class manages the beta schedule for the noise in the diffusion process. It supports various schedules
    including cosine, linear, quadratic, JSD (Jensen-Shannon Divergence), and sigmoid.

    Args:
        beta_schedule (str): The type of beta schedule to use. Supported values are "cosine", "linear",
                             "quadratic", "jsd", and "sigmoid".
        timesteps (int): The number of timesteps in the diffusion process.
        loss_type (str): The type of loss to use in the diffusion process.
        p2_loss_weight_gamma (float): Weight parameter for the P2 loss, default is 0.
        p2_loss_weight_k (int): Another weight parameter for the P2 loss, default is 1.

    Raises:
        NotImplementedError: If an unsupported beta schedule type is provided.
    """

    def __init__(self, 
                 *, 
                 beta_schedule: str, 
                 timesteps: int, 
                 loss_type: str, 
                 p2_loss_weight_gamma: float = 0.0, 
                 p2_loss_weight_k: int = 1):
        super().__init__()

        # Initialize the beta schedule based on the provided type
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError(f"Unsupported beta schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # register buffer helper function to cast double back to float

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # p2 loss reweighting

        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def sample_random_times(self, batch: int) -> torch.Tensor:
        """
        Samples random timesteps for the diffusion process.

        Args:
            batch (int): The batch size for which to generate random timesteps.

        Returns:
            torch.Tensor: A tensor of random timesteps for each element in the batch.
        """
        # Sample random timesteps uniformly for the given batch size
        return torch.randint(0, self.num_timesteps, (batch,), device=self.betas.device, dtype=torch.long)

    def q_posterior(self, 
                    x_start: torch.Tensor, 
                    x_t: torch.Tensor, 
                    t: torch.Tensor) -> tuple:
        """
        Calculates the posterior distribution q(x_{t-1} | x_t, x_0) for the diffusion process.

        Args:
            x_start (torch.Tensor): The starting point (x_0) of the diffusion process.
            x_t (torch.Tensor): The current state (x_t) of the diffusion process.
            t (torch.Tensor): The current timestep in the diffusion process.

        Returns:
            tuple: A tuple containing the mean, variance, and log variance of the posterior distribution.
        """
        # Calculate the mean of the posterior distribution
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        # Extract the variance and log variance of the posterior distribution
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, 
                 x_start: torch.Tensor, 
                 t: torch.Tensor, 
                 noise: torch.Tensor = None) -> torch.Tensor:
        """
        Samples from the diffusion process q(x_t | x_start).

        This function applies the diffusion process to the starting point (x_start) by adding scaled noise,
        following the diffusion equation based on the given timesteps.

        Args:
            x_start (torch.Tensor): The starting point (x_0) of the diffusion process.
            t (torch.Tensor): The timesteps at which to sample.
            noise (torch.Tensor, optional): The noise to add to the diffusion process. If not provided, random noise is generated.

        Returns:
            torch.Tensor: The resulting tensor after applying the diffusion process.
        """
        # Generate random noise if not provided
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Apply the diffusion process to x_start using the calculated alphas and betas
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def calculate_v(self, 
                    x_start: torch.Tensor, 
                    t: torch.Tensor, 
                    noise: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the velocity (v) in the reverse diffusion process.

        This function is used to calculate the velocity component in the reverse diffusion process,
        which is part of predicting x_start from x_t.

        Args:
            x_start (torch.Tensor): The starting point (x_0) of the diffusion process.
            t (torch.Tensor): The timesteps at which to calculate v.
            noise (torch.Tensor, optional): The noise used in the diffusion process. If not provided, random noise is generated.

        Returns:
            torch.Tensor: The calculated velocity at each timestep.
        """
        # Generate random noise if not provided
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Calculate the velocity component in the reverse diffusion process
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def q_sample_from_to(self, 
                         x_from: torch.Tensor, 
                         from_t: torch.Tensor, 
                         to_t: torch.Tensor, 
                         noise: torch.Tensor = None) -> torch.Tensor:
        """
        Samples from the diffusion process q(x_to | x_from) between two timesteps.

        This function applies the diffusion process to transition from one state of the process (x_from)
        at a given timestep (from_t) to another state at a different timestep (to_t), using the diffusion equation.

        Args:
            x_from (torch.Tensor): The starting state (x_from) of the diffusion process at from_t.
            from_t (torch.Tensor): The starting timestep for the diffusion process.
            to_t (torch.Tensor): The target timestep for the diffusion process.
            noise (torch.Tensor, optional): The noise to add to the diffusion process. If not provided, random noise is generated.

        Returns:
            torch.Tensor: The resulting tensor after applying the diffusion process from from_t to to_t.
        """
        shape = x_from.shape
        # Generate random noise if not provided
        noise = default(noise, lambda: torch.randn_like(x_from))

        # Extract alpha and sigma values for the given timesteps
        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)

        # Apply the diffusion process to transition between the timesteps
        return x_from * (alpha_next / alpha) + noise * (sigma_next * alpha - sigma * alpha_next) / alpha
    
    
    def predict_start_from_v(self, 
                             x_t: torch.Tensor, 
                             t: torch.Tensor, 
                             v: torch.Tensor) -> torch.Tensor:
        """
        Predicts the start of the diffusion process from a given state and velocity.

        This function is used in the reverse diffusion process to predict the starting point of the diffusion (x_0)
        given the current state (x_t) and the velocity (v) at a specific timestep.

        Args:
            x_t (torch.Tensor): The current state (x_t) of the diffusion process.
            t (torch.Tensor): The current timestep in the diffusion process.
            v (torch.Tensor): The velocity at the current timestep.

        Returns:
            torch.Tensor: The predicted starting point of the diffusion process.
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_start_from_noise(self, 
                                 x_t: torch.Tensor, 
                                 t: torch.Tensor, 
                                 noise: torch.Tensor) -> torch.Tensor:
        """
        Predicts the start of the diffusion process from a given state and noise.

        This function calculates the starting point of the diffusion (x_0) from the current state (x_t)
        and the applied noise at a specific timestep.

        Args:
            x_t (torch.Tensor): The current state (x_t) of the diffusion process.
            t (torch.Tensor): The current timestep in the diffusion process.
            noise (torch.Tensor): The noise applied at the current timestep.

        Returns:
            torch.Tensor: The predicted starting point of the diffusion process.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, 
                                 x_t: torch.Tensor, 
                                 t: torch.Tensor, 
                                 x0: torch.Tensor) -> torch.Tensor:
        """
        Predicts the noise applied during the diffusion process given the start and current state.

        This function calculates the noise that would have been applied to transition from the starting point (x_0)
        to the current state (x_t) of the diffusion process at a specific timestep.

        Args:
            x_t (torch.Tensor): The current state (x_t) of the diffusion process.
            t (torch.Tensor): The current timestep in the diffusion process.
            x0 (torch.Tensor): The starting point (x_0) of the diffusion process.

        Returns:
            torch.Tensor: The predicted noise applied during the diffusion process.
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / 
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def p2_reweigh_loss(self, loss, times):
        if not self.has_p2_loss_reweighting:
            return loss
        return loss * extract(self.p2_loss_weight, times, loss.shape)