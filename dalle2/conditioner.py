""" Contribution: https://github.com/lucidrains/DALLE2-pytorch """

import torch.nn as nn
from dalle2.utils import exists, default, cast_tuple, resize_image_to, identity
from dalle2.scehduler import NoiseScheduler
from kornia.filters import gaussian_blur2d
import torch
import random
from typing import Optional, Tuple, Union, Callable

class LowresConditioner(nn.Module):
    """
    Conditioning images through downsampling, blurring, and optional noise addition.

    This module is designed to process images by downsampling, applying Gaussian blur, and optionally
    adding noise. It is useful in image generation tasks where conditioning on lower-resolution images
    is beneficial.

    Attributes:
        downsample_first (bool): Whether to downsample before applying other transformations.
        use_blur (bool): Whether to apply Gaussian blur.
        blur_prob (float): Probability of applying blur.
        blur_sigma (float or tuple): Sigma value(s) for Gaussian blur.
        blur_kernel_size (int or tuple): Kernel size(s) for Gaussian blur.
        use_noise (bool): Whether to apply noise.
        input_image_range (Optional[Tuple[float, float]]): Range for clamping the image values.
        normalize_img (Callable): Function to normalize images.
        unnormalize_img (Callable): Function to unnormalize images.
        noise_scheduler (Optional[NoiseScheduler]): Scheduler for noise addition.
    """

    def __init__(
        self,
        downsample_first: bool = True,
        use_blur: bool = True,
        blur_prob: float = 0.5,
        blur_sigma: Union[float, Tuple[float, float]] = 0.6,
        blur_kernel_size: Union[int, Tuple[int, int]] = 3,
        use_noise: bool = False,
        input_image_range: Optional[Tuple[float, float]] = None,
        normalize_img_fn: Callable = lambda x: x,
        unnormalize_img_fn: Callable = lambda x: x
    ):
        super().__init__()
        self.downsample_first = downsample_first
        self.input_image_range = input_image_range

        self.use_blur = use_blur
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.blur_kernel_size = blur_kernel_size

        self.use_noise = use_noise
        self.normalize_img = normalize_img_fn
        self.unnormalize_img = unnormalize_img_fn
        self.noise_scheduler = NoiseScheduler(beta_schedule='linear', timesteps=1000, loss_type='l2') if use_noise else None

    def noise_image(self, cond_fmap: torch.Tensor, noise_levels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert exists(self.noise_scheduler)

        batch = cond_fmap.shape[0]
        cond_fmap = self.normalize_img(cond_fmap)

        random_noise_levels = default(noise_levels, lambda: self.noise_scheduler.sample_random_times(batch))
        cond_fmap = self.noise_scheduler.q_sample(cond_fmap, t=random_noise_levels, noise=torch.randn_like(cond_fmap))

        cond_fmap = self.unnormalize_img(cond_fmap)
        return cond_fmap, random_noise_levels

    def forward(
        self,
        cond_fmap: torch.Tensor,
        *,
        target_image_size: Tuple[int, int],
        downsample_image_size: Optional[Tuple[int, int]] = None,
        should_blur: bool = True,
        blur_sigma: Optional[Union[float, Tuple[float, float]]] = None,
        blur_kernel_size: Optional[Union[int, Tuple[int, int]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.downsample_first and exists(downsample_image_size):
            cond_fmap = resize_image_to(cond_fmap, downsample_image_size, clamp_range=self.input_image_range, nearest=True)

        if self.use_blur and should_blur and random.random() < self.blur_prob:
            blur_sigma = default(blur_sigma, self.blur_sigma)
            blur_kernel_size = default(blur_kernel_size, self.blur_kernel_size)
            cond_fmap = gaussian_blur2d(cond_fmap, cast_tuple(blur_kernel_size, 2), cast_tuple(blur_sigma, 2))

        cond_fmap = resize_image_to(cond_fmap, target_image_size, clamp_range=self.input_image_range, nearest=True)
        
        random_noise_levels = None
        if self.use_noise:
            cond_fmap, random_noise_levels = self.noise_image(cond_fmap)

        return cond_fmap, random_noise_levels