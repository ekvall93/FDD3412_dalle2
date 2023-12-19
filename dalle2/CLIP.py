""" Contribution: https://github.com/lucidrains/DALLE2-pytorch """

from torch import nn
from dalle2.utils import resize_image_to, l2norm
import open_clip
import torch
import torch.nn.functional as F
from typing import Any, Optional
from collections import namedtuple

EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings'])
EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])

class Clip(nn.Module):
    """
    An abstract base class for a CLIP (Contrastive Language–Image Pretraining) model.

    Attributes:
        clip (Any): The core CLIP model or its configuration.
        overrides (dict): Additional configuration overrides for the CLIP model.
    """

    def __init__(self, clip: Any, **kwargs):
        """
        Initializes the Clip class.

        Args:
            clip (Any): The core CLIP model or its configuration.
            **kwargs: Additional configuration overrides for the CLIP model.
        """
        super().__init__()
        self.clip = clip
        self.overrides = kwargs

    def validate_and_resize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Validates and resizes the input image to the required size for the CLIP model.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The resized image tensor.

        Raises:
            AssertionError: If the input image size is smaller than the required CLIP image size.
        """
        image_size = image.shape[-1]
        assert image_size >= self.image_size, f'You are passing in an image of size {image_size}, but CLIP requires the image size to be at least {self.image_size}.'
        return resize_image_to(image, self.image_size)  # Assuming resize_image_to is a predefined function.

    @property
    def dim_latent(self) -> int:
        """
        Returns the dimension of the latent space of the model.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def image_size(self) -> int:
        """
        Returns the required image size for the model.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def image_channels(self) -> int:
        """
        Returns the number of image channels used by the model.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def max_text_len(self) -> int:
        """
        Returns the maximum text length that can be processed by the model.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embeds text into a latent space using the model.

        Args:
            text (str): The input text to be embedded.

        Returns:
            torch.Tensor: The text embeddings.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Embeds an image into a latent space using the model.

        Args:
            image (torch.Tensor): The input image to be embedded.

        Returns:
            torch.Tensor: The image embeddings.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError
    
    
class OpenClip(Clip):
    """
    An implementation of the Open Clip model.
    
    """

    def __init__(self, name: str = 'hf-hub:wisdomik/QuiltNet-B-32', pretrained: Optional[Any] = None):
        """
        Initializes the OpenClip model.

        Args:
            name (str, optional): The name of the CLIP model. Default: 'hf-hub:wisdomik/QuiltNet-B-32'.
            pretrained (Optional[Any], optional): Pretrained model weights. Default: None.
        """
        clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)

        super().__init__(clip)
        self.eos_id = 49407

        text_attention_final = self.find_layer('ln_final')
        self._dim_latent = text_attention_final.weight.shape[0]

        self.handle = text_attention_final.register_forward_hook(self._hook)
        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    def find_layer(self, layer: str) -> nn.Module:
        """
        Finds a specific layer in the CLIP model.

        Args:
            layer (str): The name of the layer to find.

        Returns:
            nn.Module: The specified layer in the CLIP model.
        """
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        """
        Clears the hook registered for the model.
        """
        if not self.cleared:
            self.handle.remove()
            self.cleared = True

    def _hook(self, _, __, outputs):
        """
        A hook function to be called by PyTorch during the forward pass.
        """
        self.text_encodings = outputs

    @property
    def dim_latent(self) -> int:
        return self._dim_latent

    @property
    def image_size(self) -> int:
        image_size = self.clip.visual.image_size
        if isinstance(image_size, tuple):
            return max(image_size)
        return image_size

    @property
    def image_channels(self) -> int:
        return 3

    @property
    def max_text_len(self) -> int:
        return self.clip.context_length

    @torch.no_grad()
    def embed_text(self, text: torch.Tensor) -> EmbeddedText:
        """
        Embeds text using the CLIP model.

        Args:
            text (torch.Tensor): The input text tensor.

        Returns:
            EmbeddedText: The embeddings for the input text.
        """
        text = text[..., :self.max_text_len]

        is_eos_id = (text == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim=-1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value=True)
        text_mask = text_mask & (text != 0)
        assert not self.cleared

        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        del self.text_encodings
        return EmbeddedText(l2norm(text_embed.float()), text_encodings.float())

    @torch.no_grad()
    def embed_image(self, image: torch.Tensor) -> EmbeddedImage:
        """
        Embeds an image using the CLIP model.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            EmbeddedImage: The embeddings for the input image.
        """
        assert not self.cleared
        image = self.validate_and_resize_image(image)
        image = self.clip_normalize(image)
        image_embed = self.clip.encode_image(image)
        return EmbeddedImage(l2norm(image_embed.float()), None)
