""" Contribution: https://github.com/lucidrains/DALLE2-pytorch """

import torch
from torch import nn
import torchvision.transforms as T

from dalle2.tokenizer import tokenizer
from dalle2.CLIP import *
from dalle2.utils import eval_decorator
from dalle2.diffusion_prior import DiffusionPrior
from dalle2.utils import *
from dalle2.decoder import Decoder
from typing import Union, List

class DALLE2(nn.Module):
    """
    A PyTorch module representing the DALL-E 2 model.

    This model consists of a prior and a decoder. The prior is used to generate embeddings from text input,
    and the decoder generates images from these embeddings.

    Attributes:
        prior (DiffusionPrior): The prior model for text-to-embedding generation.
        decoder (Decoder): The decoder model for embedding-to-image generation.
        prior_num_samples (int): Number of samples to generate per batch in the prior.
        decoder_need_text_cond (bool): Flag indicating if the decoder needs text conditioning.
        to_pil (Callable): Function to convert tensors to PIL images.
    """

    def __init__(
        self,
        *,
        prior: DiffusionPrior,
        decoder: Decoder,
        prior_num_samples: int = 2
    ):
        super().__init__()
        assert isinstance(prior, DiffusionPrior)
        assert isinstance(decoder, Decoder)
        self.prior = prior
        self.decoder = decoder

        self.prior_num_samples = prior_num_samples
        self.decoder_need_text_cond = decoder.condition_on_text_encodings

        self.to_pil = T.ToPILImage()

    @eval_decorator
    def forward(
        self,
        text: Union[str, List[str]],
        cond_scale: float = 1.,
        prior_cond_scale: float = 1.,
        return_pil_images: bool = False
    ) -> Union[List[torch.Tensor], List]:
        """
        Generates images from text descriptions using the DALL-E 2 model.

        Args:
            text (Union[str, List[str]]): Text input or list of text inputs for image generation.
            cond_scale (float, optional): Conditioning scale for the decoder. Default: 1.
            prior_cond_scale (float, optional): Conditioning scale for the prior. Default: 1.
            return_pil_images (bool, optional): If True, returns images as PIL images. Default: False.

        Returns:
            Union[List[torch.Tensor], List]: List of generated images as tensors or PIL images.
        """
        device = module_device(self)
        one_text = isinstance(text, str) or (not is_list_str(text) and text.shape[0] == 1)

        if isinstance(text, str) or is_list_str(text):
            text = [text] if not isinstance(text, (list, tuple)) else text
            text = tokenizer.tokenize(text).to(device)

        image_embed = self.prior.sample(text, num_samples_per_batch=self.prior_num_samples, cond_scale=prior_cond_scale)

        text_cond = text if self.decoder_need_text_cond else None
        images = self.decoder.sample(image_embed=image_embed, text=text_cond, cond_scale=cond_scale)

        if return_pil_images:
            images = list(map(self.to_pil, images.unbind(dim=0)))

        if one_text:
            return first(images)

        return images