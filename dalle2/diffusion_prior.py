""" Contribution: https://github.com/lucidrains/DALLE2-pytorch """

from torch import nn, einsum
from einops.layers.torch import Rearrange
from dalle2.utils import exists, default, prob_mask_like
import torch 
from dalle2.MLP import MLP
from einops import rearrange
from dalle2.causal_transformer import CausalTransformer
from torch.nn import functional as F
from einops import repeat
from dalle2.scehduler import NoiseScheduler
from dalle2.utils import l2norm, freeze_model_and_make_eval_, eval_decorator
from dalle2.CLIP import Clip
from tqdm import tqdm
import random
from dalle2.causal_transformer import SinusoidalPosEmb

    
class DiffusionPriorNetwork(nn.Module):
    """
    Diffusion Prior Network for generating embeddings for text, images, and time.

    Attributes:
        dim (int): Dimensionality of the embeddings.
        num_timesteps (Optional[int]): Number of timesteps for embedding generation. None for continuous embedding.
        num_time_embeds (int): Number of embeddings for time.
        num_image_embeds (int): Number of embeddings for images.
        num_text_embeds (int): Number of embeddings for text.
        max_text_len (int): Maximum length of text for embeddings.
        self_cond (bool): Flag to use self conditioning technique.

    Args:
        dim (int): Dimensionality of the embeddings.
        num_timesteps (Optional[int]): Number of timesteps, default is None.
        num_time_embeds (int): Number of time embeddings, default is 1.
        num_image_embeds (int): Number of image embeddings, default is 1.
        num_text_embeds (int): Number of text embeddings, default is 1.
        max_text_len (int): Maximum length of text for embeddings, default is 256.
        self_cond (bool): Flag to use self conditioning, default is False.
    """

    def __init__(
        self,
        dim: int,
        num_timesteps: int = None,
        num_time_embeds: int = 1,
        num_image_embeds: int = 1,
        num_text_embeds: int = 1,
        max_text_len: int = 256,
        self_cond: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        self.num_time_embeds = num_time_embeds
        self.num_image_embeds = num_image_embeds
        self.num_text_embeds = num_text_embeds

        # Create text embeddings
        self.to_text_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_text_embeds) if num_text_embeds > 1 else nn.Identity(),
            Rearrange('b (n d) -> b n d', n=num_text_embeds)
        )

        # Determine if continuous time embedding is used
        self.continuous_embedded_time = num_timesteps is None

        # Create time embeddings
        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if num_timesteps is not None else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)),
            Rearrange('b (n d) -> b n d', n=num_time_embeds)
        )

        # Create image embeddings
        self.to_image_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_image_embeds) if num_image_embeds > 1 else nn.Identity(),
            Rearrange('b (n d) -> b n d', n=num_image_embeds)
        )

        self.learned_query = nn.Parameter(torch.randn(dim))
        self.causal_transformer = CausalTransformer(dim=dim, **kwargs)

        # Initialize null encodings for text and images
        self.max_text_len = max_text_len
        self.null_text_encodings = nn.Parameter(torch.randn(1, max_text_len, dim))
        self.null_text_embeds = nn.Parameter(torch.randn(1, num_text_embeds, dim))
        self.null_image_embed = nn.Parameter(torch.randn(1, dim))

        self.self_cond = self_cond  # Flag for self conditioning

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with conditional scaling applied to the logits.

        This method performs a forward pass using the provided arguments and applies a conditional scaling factor to the logits. 
        It is designed to adjust the influence of certain conditions (like text or image embeddings) on the final output.

        Args:
            *args: Variable length argument list for the forward pass.
            cond_scale (float): A scaling factor for conditioning. Default is 1.0, which implies no scaling.
            **kwargs: Arbitrary keyword arguments for the forward pass.

        Returns:
            torch.Tensor: The scaled logits resulting from the forward pass.
        """

        # Perform the standard forward pass
        logits = self.forward(*args, **kwargs)

        # If the conditional scale is 1, return the original logits
        if cond_scale == 1:
            return logits

        # Compute 'null' logits where text and image conditions are dropped
        null_logits = self.forward(
            *args, 
            text_cond_drop_prob=1.0, 
            image_cond_drop_prob=1.0, 
            **kwargs
        )

        # Apply conditional scaling to the logits
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed: torch.Tensor,
        diffusion_timesteps: torch.Tensor,
        *,
        text_embed: torch.Tensor,
        text_encodings: torch.Tensor = None,
        self_cond: torch.Tensor = None,
        text_cond_drop_prob: float = 0.0,
        image_cond_drop_prob: float = 0.0
    ) -> torch.Tensor:
        """
        Performs a forward pass of the Diffusion Prior Network.

        The method processes embeddings for images, diffusion timesteps, and text, applying various masks and conditioning
        techniques. It integrates self-conditioning, text and image conditional drop probabilities, and handles the 
        padding and masking of text encodings.

        Args:
            image_embed (torch.Tensor): The image embeddings.
            diffusion_timesteps (torch.Tensor): The diffusion timesteps.
            text_embed (torch.Tensor): The text embeddings.
            text_encodings (torch.Tensor, optional): The text encodings, default is None.
            self_cond (torch.Tensor, optional): Tensor for self-conditioning, default is None.
            text_cond_drop_prob (float): Probability of dropping text conditions, default is 0.0.
            image_cond_drop_prob (float): Probability of dropping image conditions, default is 0.0.

        Returns:
            torch.Tensor: The predicted image embeddings after processing.
        """

        batch, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype

        # Process text and image embeddings with conditional dropout
        text_embed = self.to_text_embeds(text_embed)
        image_embed = self.to_image_embeds(image_embed)

        # Create masks based on conditional dropout probabilities
        text_keep_mask = prob_mask_like((batch,), 1 - text_cond_drop_prob, device=device)
        text_keep_mask = rearrange(text_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device=device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

        # Handle optional text encodings and padding
        if not exists(text_encodings):
            text_encodings = torch.empty((batch, 0, dim), device=device, dtype=dtype)
        
        mask = torch.any(text_encodings != 0., dim=-1)
        text_encodings = text_encodings[:, :self.max_text_len]
        mask = mask[:, :self.max_text_len]

        text_len = text_encodings.shape[-2]
        remainder = self.max_text_len - text_len
        if remainder > 0:
            text_encodings = F.pad(text_encodings, (0, 0, 0, remainder), value=0.)
            mask = F.pad(mask, (0, remainder), value=False)

        # Apply masks to text and image embeddings
        null_text_encodings = self.null_text_encodings.to(text_encodings.dtype)
        text_encodings = torch.where(rearrange(mask, 'b n -> b n 1').clone() & text_keep_mask, text_encodings, null_text_encodings)

        null_text_embeds = self.null_text_embeds.to(text_embed.dtype)
        text_embed = torch.where(text_keep_mask, text_embed, null_text_embeds)

        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(image_keep_mask, image_embed, null_image_embed)

        # Process diffusion timesteps
        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        # Concatenate tokens and perform transformer attention
        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b=batch)
        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros(batch, self.dim, device=device, dtype=dtype))
            self_cond = rearrange(self_cond, 'b d -> b 1 d')
            learned_queries = torch.cat((self_cond, learned_queries), dim=-2)

        tokens = torch.cat((text_encodings, text_embed, time_embed, image_embed, learned_queries), dim=-2)
        tokens = self.causal_transformer(tokens)

        # Extract and return the predicted image embedding
        pred_image_embed = tokens[..., -1, :]
        return pred_image_embed

class DiffusionPrior(nn.Module):
    """
    Diffusion Prior class for setting up a diffusion model with various configurations.

    This class is designed to integrate a neural network with CLIP embeddings and diffusion models, supporting various
    options like conditional dropout probabilities, beta scheduling, and loss types. It's tailored for tasks like 
    generating images from text embeddings, where the text is processed through CLIP and the image generation follows 
    a diffusion process.

    Args:
        net (nn.Module): The neural network to be used in the diffusion process.
        clip (Optional[nn.Module]): The CLIP model used for text and image processing, default is None.
        image_embed_dim (Optional[int]): The dimension of the image embeddings, default is None.
        image_size (Optional[int]): The size of the images to be generated, default is None.
        image_channels (int): The number of channels in the images, default is 3.
        timesteps (int): The number of timesteps in the diffusion process, default is 1000.
        sample_timesteps (Optional[int]): Specific timesteps for sampling, default is None.
        cond_drop_prob (float): The probability of dropping conditions in general, default is 0.0.
        text_cond_drop_prob (Optional[float]): Specific dropout probability for text conditions, default is None.
        image_cond_drop_prob (Optional[float]): Specific dropout probability for image conditions, default is None.
        loss_type (str): The type of loss to use, e.g., "l2", default is "l2".
        predict_x_start (bool): Flag to predict the start of the process, default is True.
        predict_v (bool): Flag to predict the velocity in the diffusion process, default is False.
        beta_schedule (str): The type of beta schedule to use, e.g., "cosine", default is "cosine".
        condition_on_text_encodings (bool): Whether to condition on text encodings, default is True.
        sampling_clamp_l2norm (bool): Whether to apply L2 norm clamping during sampling, default is False.
        sampling_final_clamp_l2norm (bool): Whether to apply L2 norm clamping on the final output, default is False.
        training_clamp_l2norm (bool): Whether to apply L2 norm clamping during training, default is False.
        init_image_embed_l2norm (bool): Whether to apply L2 norm to the initial image embedding, default is False.
        image_embed_scale (Optional[float]): Scale for the L2-normed image embedding, default is None.
        clip_adapter_overrides (dict): Overrides for the CLIP adapter, default is an empty dict.

    Attributes:
        sample_timesteps (Optional[int]): Specific timesteps for sampling.
        noise_scheduler (NoiseScheduler): The scheduler for noise levels in the diffusion process.
    """

    def __init__(
        self,
        net: nn.Module,
        *,
        clip: nn.Module = None,
        image_embed_dim: int = None,
        image_size: int = None,
        image_channels: int = 3,
        timesteps: int = 1000,
        sample_timesteps: int = None,
        cond_drop_prob: float = 0.0,
        text_cond_drop_prob: float = None,
        image_cond_drop_prob: float = None,
        loss_type: str = "l2",
        predict_x_start: bool = True,
        predict_v: bool = False,
        beta_schedule: str = "cosine",
        condition_on_text_encodings: bool = True,
        sampling_clamp_l2norm: bool = False,
        sampling_final_clamp_l2norm: bool = False,
        training_clamp_l2norm: bool = False,
        init_image_embed_l2norm: bool = False,
        image_embed_scale: float = None,
        clip_adapter_overrides: dict = {}
    ):
        super().__init__()

        self.sample_timesteps = sample_timesteps

        # Initialize the noise scheduler with the specified beta schedule, timesteps, and loss type
        self.noise_scheduler = NoiseScheduler(
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            loss_type=loss_type
        )

        # Validate and initialize CLIP model if provided
        if exists(clip):
            assert image_channels == clip.image_channels, \
                f'channels of image ({image_channels}) should be equal to the channels that CLIP accepts ({clip.image_channels})'

            assert isinstance(clip, Clip)  # Ensure clip is an instance of the Clip class
            freeze_model_and_make_eval_(clip)  # Freeze the CLIP model and set it to evaluation mode
            self.clip = clip
        else:
            # If no CLIP model is provided, ensure that the image embedding dimension is specified
            assert exists(image_embed_dim), 'latent dimension must be given, if training prior network without CLIP given'
            self.clip = None

        # Initialize the neural network and set image embedding dimensions
        self.net = net
        self.image_embed_dim = default(image_embed_dim, lambda: clip.dim_latent)
        assert net.dim == self.image_embed_dim, \
            f'your diffusion prior network has a dimension of {net.dim}, but you set your image embedding dimension (keyword image_embed_dim) on DiffusionPrior to {self.image_embed_dim}'
        assert not exists(clip) or clip.dim_latent == self.image_embed_dim, \
            f'you passed in a CLIP to the diffusion prior with latent dimensions of {clip.dim_latent}, but your image embedding dimension (keyword image_embed_dim) for the DiffusionPrior was set to {self.image_embed_dim}'

        # Set default values for image channels and conditional drop probabilities
        self.channels = default(image_channels, lambda: clip.image_channels)
        self.text_cond_drop_prob = default(text_cond_drop_prob, cond_drop_prob)
        self.image_cond_drop_prob = default(image_cond_drop_prob, cond_drop_prob)

        # Determine if classifier guidance is possible
        self.can_classifier_guidance = self.text_cond_drop_prob > 0. and self.image_cond_drop_prob > 0.
        self.condition_on_text_encodings = condition_on_text_encodings

        # Select prediction method for diffusion process
        self.predict_x_start = predict_x_start
        self.predict_v = predict_v  # Takes precedence over predict_x_start

        # Set image embedding scaling based on a suggestion by @crowsonkb
        self.image_embed_scale = default(image_embed_scale, self.image_embed_dim ** 0.5)

        # Set flags for L2 norm clamping
        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.sampling_final_clamp_l2norm = sampling_final_clamp_l2norm
        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

        # Register a buffer for tracking the device
        self.register_buffer('_dummy', torch.tensor([True]), persistent=False)

    @property
    def device(self):
        """
        Property to get the device of the model.

        This property returns the device of the internal tensor '_dummy'. It's a convenient way to
        identify the device (CPU or GPU) on which the model is loaded.

        Returns:
            torch.device: The device on which the model is operating.
        """
        return self._dummy.device

    def l2norm_clamp_embed(self, image_embed: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 norm clamping to the image embeddings.

        This method normalizes the image embeddings to a unit vector and then scales it by the image embed scale.
        This process is used to keep the embeddings within a reasonable range during the diffusion process.

        Args:
            image_embed (torch.Tensor): The image embeddings to be clamped.

        Returns:
            torch.Tensor: The L2 norm clamped image embeddings.
        """
        # Normalize the image embeddings to a unit vector and scale
        return l2norm(image_embed) * self.image_embed_scale

    def p_mean_variance(self, 
                    x: torch.Tensor, 
                    t: torch.Tensor, 
                    text_cond, 
                    self_cond: torch.Tensor = None, 
                    clip_denoised: bool = False, 
                    cond_scale: float = 1.0) -> tuple:
        """
        Calculates the mean and variance of the posterior distribution for the diffusion process.

        This method predicts the starting point of the diffusion process (x_start) and calculates the posterior
        mean and variance. It supports different options for predicting x_start, such as predicting from noise,
        predicting directly, or using velocity. The method also supports classifier-free guidance by scaling the 
        conditions.

        Args:
            x (torch.Tensor): The current state of the diffusion process.
            t (torch.Tensor): The current timesteps in the diffusion process.
            text_cond: The text condition inputs.
            self_cond (torch.Tensor, optional): The self-conditioning tensor, default is None.
            clip_denoised (bool): Flag to clip the denoised image in the range [-1, 1], default is False.
            cond_scale (float): The scale for classifier-free guidance, default is 1.0.

        Returns:
            tuple: A tuple containing the model mean, posterior variance, posterior log variance, and x_start.

        Raises:
            AssertionError: If conditional scale is not 1 and model wasn't trained with conditional dropout.
        """
        # Ensure proper usage of classifier free guidance
        assert not (cond_scale != 1. and not self.can_classifier_guidance), \
            'the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        # Predict the starting point of the diffusion process with conditional scaling
        pred = self.net.forward_with_cond_scale(x, t, cond_scale=cond_scale, self_cond=self_cond, **text_cond)

        # Choose the method to predict x_start based on model configuration
        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        # Optionally clamp x_start in the range [-1, 1]
        if clip_denoised and not self.predict_x_start:
            x_start.clamp_(-1., 1.)

        # Optionally apply L2 norm clamping if x_start is predicted and sampling clamp is enabled
        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.image_embed_scale

        # Calculate the mean, variance, and log variance of the posterior distribution
        model_mean, posterior_variance, posterior_log_variance = self.noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                text_cond = None, 
                self_cond: torch.Tensor = None, 
                clip_denoised: bool = True, 
                cond_scale: float = 1.0) -> tuple:
        """
        Samples from the model at a given timestep.

        This method performs a sampling step in the diffusion process. It calculates the mean, variance, and
        starting point of the diffusion process and then uses this information to sample the next state.
        The sampling accounts for the presence of noise, which is omitted at timestep 0.

        Args:
            x (torch.Tensor): The current state of the diffusion process.
            t (torch.Tensor): The current timesteps in the diffusion process.
            text_cond: The text condition inputs.
            self_cond (torch.Tensor, optional): The self-conditioning tensor, default is None.
            clip_denoised (bool): Flag to clip the denoised image in the range [-1, 1], default is True.
            cond_scale (float): The scale for classifier-free guidance, default is 1.0.

        Returns:
            tuple: A tuple containing the predicted next state and the starting point of the diffusion process.
        """
        b, *_, device = *x.shape, x.device

        # Calculate the mean, variance, and starting point of the diffusion process
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=t, text_cond=text_cond, self_cond=self_cond, 
            clip_denoised=clip_denoised, cond_scale=cond_scale
        )

        # Generate random noise
        noise = torch.randn_like(x)

        # Mask to ensure no noise is added when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # Predict the next state by adding scaled noise to the model mean
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(self, 
                        shape: tuple, 
                        text_cond, 
                        cond_scale: float = 1.0) -> torch.Tensor:
        """
        Performs a sampling loop using the DDPM (Denoising Diffusion Probabilistic Models) approach.

        This method iteratively samples from the model across all timesteps, starting from a random noise image
        embedding and progressively denoising it to generate an image. The method supports self-conditioning and
        applies L2 norm clamping as specified in the model configuration.

        Args:
            shape (tuple): The shape of the tensor to be sampled (batch size, channels, height, width).
            text_cond: The text condition inputs.
            cond_scale (float): The scale for classifier-free guidance, default is 1.0.

        Returns:
            torch.Tensor: The generated image embedding after the sampling loop.

        Note:
            This method uses tqdm for progress visualization, which is helpful in long sampling loops.
        """
        batch, device = shape[0], self.device

        # Initialize image embeddings with random noise
        image_embed = torch.randn(shape, device=device)
        x_start = None  # for self-conditioning

        # Apply L2 norm clamping to the initial image embedding if configured
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        # Iterate over timesteps in reverse order for sampling
        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps):
            times = torch.full((batch,), i, device=device, dtype=torch.long)

            # Determine if self-conditioning is used
            self_cond = x_start if self.net.self_cond else None

            # Sample from the model at each timestep
            image_embed, x_start = self.p_sample(image_embed, times, text_cond=text_cond, self_cond=self_cond, cond_scale=cond_scale)

        # Apply final L2 norm clamping if configured
        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddim(self, 
                        shape: tuple, 
                        text_cond, 
                        *, 
                        timesteps: int, 
                        eta: float = 1.0, 
                        cond_scale: float = 1.0) -> torch.Tensor:
        """
        Performs a sampling loop using the DDIM (Denoising Diffusion Implicit Models) approach.

        This method iteratively samples from the model using a non-Markovian process. It starts from a random
        noise image embedding and progressively denoises it to generate an image, following the DDIM sampling
        approach with specified timesteps.

        Args:
            shape (tuple): The shape of the tensor to be sampled (batch size, channels, height, width).
            text_cond: The text condition inputs.
            timesteps (int): The number of timesteps to use for the sampling process.
            eta (float): The eta parameter for controlling noise in the DDIM process, default is 1.0.
            cond_scale (float): The scale for classifier-free guidance, default is 1.0.

        Returns:
            torch.Tensor: The generated image embedding after the DDIM sampling loop.

        Note:
            This method uses tqdm for progress visualization, which is helpful in long sampling loops.
        """
        batch, device, alphas, total_timesteps = shape[0], self.device, self.noise_scheduler.alphas_cumprod_prev, self.noise_scheduler.num_timesteps

        # Define time steps for DDIM process
        times = torch.linspace(-1., total_timesteps, steps=timesteps + 1)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Initialize image embeddings with random noise
        image_embed = torch.randn(shape, device=device)
        x_start = None  # for self-conditioning

        # Apply L2 norm clamping to the initial image embedding if configured
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        # Iterate over time pairs for DDIM sampling
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            alpha, alpha_next = alphas[time], alphas[time_next]
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            # Determine if self-conditioning is used
            self_cond = x_start if self.net.self_cond else None

            # Predict the next state
            pred = self.net.forward_with_cond_scale(image_embed, time_cond, self_cond=self_cond, cond_scale=cond_scale, **text_cond)

            # Derive x0 (start of the diffusion process)
            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(image_embed, t=time_cond, v=pred)
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(image_embed, t=time_cond, noise=pred)

            # Optionally clamp x_start and apply L2 norm clamping
            if not self.predict_x_start:
                x_start.clamp_(-1., 1.)
            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            # Predict noise for the next state
            pred_noise = self.noise_scheduler.predict_noise_from_start(image_embed, t=time_cond, x0=x_start)
            if time_next < 0:
                image_embed = x_start
                continue

            # Update image embedding based on DDIM equation
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(image_embed) if time_next > 0 else 0.
            image_embed = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise

        # Apply final L2 norm clamping if configured
        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps: int = None, **kwargs) -> torch.Tensor:
        """
        Chooses and performs the appropriate sampling loop based on the specified timesteps.

        This method decides between using the DDPM or DDIM sampling loop based on the number of timesteps. It
        normalizes the resulting image embedding by dividing it by the image embedding scale.

        Args:
            *args: Variable length argument list, typically includes inputs to the sampling method.
            timesteps (int, optional): The number of timesteps to use for the sampling process. If not specified,
                                    defaults to the total number of timesteps in the noise scheduler.
            **kwargs: Arbitrary keyword arguments, typically includes additional parameters for the sampling method.

        Returns:
            torch.Tensor: The resulting image embedding after sampling and normalization.

        Note:
            If the specified timesteps are less than the total timesteps in the noise scheduler, DDIM sampling
            is used. Otherwise, DDPM sampling is used.
        """
        # Set default timesteps if not provided
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps, "Specified timesteps exceed the configured total timesteps."

        # Determine whether to use DDIM or DDPM based on the number of timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        # Perform the appropriate sampling loop
        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, timesteps=timesteps, **kwargs)

        # Normalize the resulting image embedding
        image_embed = normalized_image_embed / self.image_embed_scale
        return image_embed

    def p_losses(self, 
             image_embed: torch.Tensor, 
             times: torch.Tensor, 
             text_cond, 
             noise: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the loss for the diffusion process.

        This method applies noise to the image embeddings, optionally adds self-conditioning, and predicts
        the target (either noise, x_start, or v). It then calculates the loss between the prediction and the target.

        Args:
            image_embed (torch.Tensor): The image embeddings.
            times (torch.Tensor): The timesteps at which the embeddings are noised.
            text_cond: The text condition inputs.
            noise (torch.Tensor, optional): The noise to be added to the image embeddings. If not provided, random noise is generated.

        Returns:
            torch.Tensor: The calculated loss for the given inputs.

        Note:
            The method supports three types of targets based on the model configuration: noise (default), x_start, and v.
            It also supports optional self-conditioning and L2 norm clamping during training.
        """
        # Generate random noise if not provided
        noise = default(noise, lambda: torch.randn_like(image_embed))

        # Apply noise to the image embeddings
        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)

        # Optionally add self-conditioning
        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        # Predict the target based on the noisy image embeddings
        pred = self.net(
            image_embed_noisy,
            times,
            self_cond=self_cond,
            text_cond_drop_prob=self.text_cond_drop_prob,
            image_cond_drop_prob=self.image_cond_drop_prob,
            **text_cond
        )

        # Optionally apply L2 norm clamping during training
        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        # Determine the target for loss calculation
        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        # Calculate and return the loss
        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss

    @torch.no_grad()
    @eval_decorator
    def sample_batch_size(self, 
                        batch_size: int, 
                        text_cond, 
                        cond_scale: float = 1.0) -> torch.Tensor:
        """
        Generates a batch of samples using the diffusion model.

        This method iteratively samples images from the model for the given batch size. It progresses through the diffusion
        timesteps in reverse, updating the image at each step based on the model's predictions.

        Args:
            batch_size (int): The number of samples to generate.
            text_cond: The text condition inputs for the model.
            cond_scale (float): The scale for classifier-free guidance, default is 1.0.

        Returns:
            torch.Tensor: A tensor containing the generated samples.

        Note:
            This method uses tqdm for progress visualization, which is helpful in long sampling loops.
        """
        # Set the device and shape for the image generation
        device = self.betas.device
        shape = (batch_size, self.image_embed_dim)

        # Initialize the image tensor with random noise
        img = torch.randn(shape, device=device)

        # Iterate over timesteps in reverse order for sampling
        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps):
            times = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, times, text_cond=text_cond, cond_scale=cond_scale)

        return img

    @torch.no_grad()
    @eval_decorator
    def sample(self, 
            text, 
            num_samples_per_batch: int = 2, 
            cond_scale: float = 1.0, 
            timesteps: int = None) -> torch.Tensor:
        """
        Generates image embeddings based on text input, selecting the top similarity as judged by CLIP.

        This method samples multiple image embeddings for each text input, uses the CLIP model to judge similarity,
        and selects the top one. It supports conditioning on both text embeddings and text encodings.

        Args:
            text: The text input for generating image embeddings.
            num_samples_per_batch (int): The number of image embeddings to sample per text input, default is 2.
            cond_scale (float): The scale for classifier-free guidance, default is 1.0.
            timesteps (int, optional): The number of timesteps to use for sampling. If not specified, defaults to the 
                                    pre-configured sample timesteps of the model.

        Returns:
            torch.Tensor: The tensor of top image embeddings corresponding to the input text.
        """
        timesteps = default(timesteps, self.sample_timesteps)

        # Replicate text for multiple samples per batch
        text = repeat(text, 'b ... -> (b r) ...', r=num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        # Get text embeddings and encodings from CLIP
        text_embed, text_encodings = self.clip.embed_text(text)
        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # Sample image embeddings
        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), text_cond=text_cond, cond_scale=cond_scale, timesteps=timesteps)

        # Calculate similarities between text and image embeddings
        text_embeds = rearrange(text_cond['text_embed'], '(b r) d -> b r d', r=num_samples_per_batch)
        image_embeds = rearrange(image_embeds, '(b r) d -> b r d', r=num_samples_per_batch)
        text_image_sims = torch.einsum('b r d, b r d -> b r', l2norm(text_embeds), l2norm(image_embeds))

        # Select top similarity indices
        top_sim_indices = text_image_sims.topk(k=1).indices
        top_sim_indices = repeat(top_sim_indices, 'b 1 -> b 1 d', d=image_embed_dim)

        # Gather top image embeddings based on similarity
        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, 'b 1 d -> b d')

    def forward(self,
            text: torch.Tensor = None,
            image: torch.Tensor = None,
            text_embed: torch.Tensor = None,
            image_embed: torch.Tensor = None,
            text_encodings: torch.Tensor = None,
            *args,
            **kwargs) -> torch.Tensor:
        """
        Forward pass for the diffusion model.

        This method processes text and/or image inputs, optionally using preprocessed CLIP embeddings and encodings,
        to compute the diffusion loss. It supports conditioning on text embeddings and text encodings.

        Args:
            text (torch.Tensor, optional): Raw text input.
            image (torch.Tensor, optional): Raw image input.
            text_embed (torch.Tensor, optional): Preprocessed CLIP text embeddings.
            image_embed (torch.Tensor, optional): Preprocessed CLIP image embeddings or embeddings to be processed.
            text_encodings (torch.Tensor, optional): Preprocessed CLIP text encodings.
            *args: Additional variable length argument list.
            **kwargs: Additional arbitrary keyword arguments.

        Returns:
            torch.Tensor: The computed loss for the input data.

        Raises:
            AssertionError: If required conditions on inputs are not met.

        Note:
            The method requires either text or text embeddings, and either image or image embeddings.
            It also asserts the presence of text encodings if the model is configured to condition on them.
        """
        # Assertions to ensure correct inputs are provided
        assert exists(text) ^ exists(text_embed), 'either text or text embedding must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), \
            'text encodings must be present if you specified you wish to condition on it on initialization'

        # Process image input
        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # Calculate text conditionings based on provided inputs
        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed=text_embed)
        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # Sample random timesteps for conditioning
        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # Scale image embedding as per configuration
        image_embed *= self.image_embed_scale

        # Calculate and return the forward loss
        return self.p_losses(image_embed, times, text_cond=text_cond, *args, **kwargs)
