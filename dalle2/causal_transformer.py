from torch import nn, einsum
from einops import rearrange, pack, unpack, repeat
import torch
import math
from dalle2.utils import exists, l2norm, is_float_dtype
from rotary_embedding_torch import RotaryEmbedding
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Any, Optional

class SinusoidalPosEmb(nn.Module):
    """
    A module for creating sinusoidal positional embeddings.

    The sinusoidal positional embedding method where the position of each element in the sequence is encoded as a sinusoid.
    The frequencies of the sinusoids are geometrically spaced.

    Attributes:
        dim (int): The dimensionality of the embeddings.
    """

    def __init__(self, dim: int):
        """
        Initializes the SinusoidalPosEmb module.

        Args:
            dim (int): The dimensionality of the embeddings.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the sinusoidal positional embeddings.

        Args:
            x (Tensor): A tensor of shape (sequence_length, ) containing the positions
                        in the sequence for which to generate embeddings.

        Returns:
            Tensor: A tensor of shape (sequence_length, dim) containing the sinusoidal
                    positional embeddings.
        """
        dtype, device = x.dtype, x.device
        assert is_float_dtype(dtype), 'input to sinusoidal pos emb must be a float type'

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1).type(dtype)
    

class RearrangeToSequence(nn.Module):
    """
    Rearranges a tensor's dimensions, processes it through a given function,
    and then rearranges it back to its original shape.

    This module is useful for applying functions that expect a specific input shape (like sequence models)
    to tensors of different shapes.

    Attributes:
        fn (Callable): The function to apply to the rearranged tensor.
    """

    def __init__(self, fn: Callable):
        """
        Initializes the RearrangeToSequence module.

        Args:
            fn (Callable): The function to apply to the rearranged tensor. This function
                           should accept a tensor as input and return a tensor.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the rearrangement and function to the input tensor.

        The input tensor is first rearranged to move the channel dimension to the end,
        processed through the function, and then rearranged back to its original shape.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor, which is the result of applying the function
                          to the input tensor after rearrangement.
        """
        # Rearrange the tensor, moving the channel dimension to the end
        x = rearrange(x, 'b c ... -> b ... c')

        # Pack the tensor for processing
        x, ps = pack([x], 'b * c')

        # Apply the function
        x = self.fn(x)

        # Unpack and rearrange back to the original shape
        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b ... c -> b c ...')

        return x
    
class LayerNorm(nn.Module):
    """
    Layer normalization.

    Used to normalize the inputs across the features, and stabilize the learning process.
    This implementation also includes an option for stable normalization and supports different
    epsilon values for different data types (e.g., float32, float16).

    Attributes:
        eps (float): Epsilon value for numerical stability in float32.
        fp16_eps (float): Epsilon value for numerical stability in float16.
        stable (bool): Whether to use stable normalization.
        g (torch.nn.Parameter): Learnable gain parameters.
    """

    def __init__(self, dim: int, eps: float = 1e-5, fp16_eps: float = 1e-3, stable: bool = False):
        """
        Initializes the LayerNorm module.

        Args:
            dim (int): The number of features in the input.
            eps (float, optional): Epsilon value for numerical stability in float32. Default: 1e-5.
            fp16_eps (float, optional): Epsilon value for numerical stability in float16. Default: 1e-3.
            stable (bool, optional): Whether to use stable normalization. Default: False.
        """
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the layer normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class ChanLayerNorm(nn.Module):
    """
    Channel-wise layer normalization.

    Channel-wise layer normalization normalize the inputs across the channels.
    This implementation supports stable normalization
    and different epsilon values for different data types (e.g., float32, float16).

    Attributes:
        eps (float): Epsilon value for numerical stability in float32.
        fp16_eps (float): Epsilon value for numerical stability in float16.
        stable (bool): Whether to use stable normalization.
        g (torch.nn.Parameter): Learnable gain parameters, shaped for channel-wise normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-5, fp16_eps: float = 1e-3, stable: bool = False):
        """
        Initializes the ChanLayerNorm module.

        Args:
            dim (int): The number of channels in the input.
            eps (float, optional): Epsilon value for numerical stability in float32. Default: 1e-5.
            fp16_eps (float, optional): Epsilon value for numerical stability in float16. Default: 1e-3.
            stable (bool, optional): Whether to use stable normalization. Default: False.
        """
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the channel-wise layer normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor, typically a 4D tensor for convolutional neural networks.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()

        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class Residual(nn.Module):
    """
    Residual connection.

    In a residual connection, the output of a function (typically a neural network layer)
    is added to its input. This technique mitigates the vanishing gradient problem and enable the training of deeper networks.

    Attributes:
        fn (Callable): The function to apply in the residual connection. This function
                       should accept a tensor as input and return a tensor.
    """

    def __init__(self, fn: Callable[..., Any]):
        """
        Initializes the Residual module.

        Args:
            fn (Callable[..., Any]): The function to apply in the residual connection. This function
                                     should accept a tensor as input and return a tensor. It can
                                     also accept additional keyword arguments.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Applies the function with residual connection to the input tensor.

        The output of the function is added to the input tensor, and the result is returned.

        Args:
            x (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            torch.Tensor: The output tensor, which is the result of the function added to the input tensor.
        """
        return self.fn(x, **kwargs) + x


class RelPosBias(nn.Module):
    """
    Relative positional bias in attention mechanisms.

    This module generates a relative positional bias based on the positions of query and key
    tokens in a sequence. This bias is used in transformer models to provide information about
    the relative positions of tokens, which is crucial for understanding sequence order and structure.

    Attributes:
        num_buckets (int): The number of buckets for discretizing relative positions.
        max_distance (int): The maximum distance to consider for relative positions.
        relative_attention_bias (torch.nn.Embedding): An embedding layer that maps relative positions
                                                      to bias values.
    """

    def __init__(self, heads: int = 8, num_buckets: int = 32, max_distance: int = 128):
        """
        Initializes the RelPosBias module.

        Args:
            heads (int, optional): The number of attention heads. Default: 8.
            num_buckets (int, optional): The number of buckets for discretizing relative positions. Default: 32.
            max_distance (int, optional): The maximum distance to consider for relative positions. Default: 128.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int = 32, max_distance: int = 128) -> torch.Tensor:
        """
        Computes the relative position bucket index for a given relative position.

        This function discretizes relative positions into fixed number of buckets, which is
        used to retrieve the corresponding bias values.

        Args:
            relative_position (torch.Tensor): The tensor of relative positions.
            num_buckets (int, optional): The number of buckets for discretization. Default: 32.
            max_distance (int, optional): The maximum distance for relative positions. Default: 128.

        Returns:
            torch.Tensor: The tensor containing bucket indices for each relative position.
        """
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, i: int, j: int, *, device: torch.device) -> torch.Tensor:
        """
        Computes the relative positional bias for given dimensions.

        Args:
            i (int): The dimension of the query sequence.
            j (int): The dimension of the key sequence.
            device (torch.device): The device on which to perform computations.

        Returns:
            torch.Tensor: The tensor containing relative positional bias for each pair of positions in the query and key sequences.
        """
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.

    SwiGLU (Sigmoid-weighted Linear Unit) is a variant of the Gated Linear Unit (GLU)
    activation function. It uses the SiLU (Sigmoid Linear Unit) as the gating mechanism.
    """

    def __init__(self):
        """
        Initializes the SwiGLU module.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SwiGLU activation function to the input tensor.

        The input tensor is split into two halves along the last dimension,
        with one half used as the gate applied to the other half after applying
        the SiLU (sigmoid linear unit) function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the SwiGLU activation function.
        """
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)
    

def FeedForward(
    dim: int,
    mult: int = 4,
    dropout: float = 0.,
    post_activation_norm: bool = False
) -> nn.Module:
    """
    Feedforward neural network layer used in transformer architectures.

    Args:
        dim (int): The dimensionality of the input.
        mult (int, optional): The multiplier for the inner dimension of the feedforward layer. Default: 4.
        dropout (float, optional): The dropout rate. Default: 0.
        post_activation_norm (bool, optional): Flag to include layer normalization after the activation function.
                                              Based on the concept from https://arxiv.org/abs/2110.09456. Default: False.

    Returns:
        nn.Module: A sequential model representing the feedforward network layer.
    """
    inner_dim = int(mult * dim)
    layers = [
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    ]

    return nn.Sequential(*layers)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        rotary_emb = None,
        cosine_sim = True,
        cosine_sim_scale = 16
    ):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # rotary embeddings

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # whether to use cosine sim

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CausalTransformer(nn.Module):
    """
    Transformer model with a causal attention mechanism.

    This transformer model is designed for tasks requiring causal attention (like autoregressive models).
    It includes options for initial and final layer normalization, relative positional bias, rotary embeddings,
    feedforward layers, and a final projection layer.

    Attributes:
        init_norm (Union[LayerNorm, nn.Identity]): Initial normalization, either LayerNorm or Identity.
        rel_pos_bias (RelPosBias): Relative positional bias for attention.
        layers (nn.ModuleList): The list of transformer layers, each containing Attention and FeedForward modules.
        norm (Union[LayerNorm, nn.Identity]): Final normalization layer.
        project_out (Union[nn.Linear, nn.Identity]): Final projection layer.
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        norm_in: bool = False,
        norm_out: bool = True,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        final_proj: bool = True,
        normformer: bool = False,
        rotary_emb: bool = True
    ):
        """
        Initializes the CausalTransformer module.

        Args:
            dim (int): Dimension of input features.
            depth (int): Number of layers in the transformer.
            dim_head (int, optional): Dimension of each attention head. Default: 64.
            heads (int, optional): Number of attention heads. Default: 8.
            ff_mult (int, optional): Multiplier for the inner dimension of the feedforward layers. Default: 4.
            norm_in (bool, optional): Whether to use initial layer normalization. Default: False.
            norm_out (bool, optional): Whether to use final layer normalization. Default: True.
            attn_dropout (float, optional): Dropout rate for attention weights. Default: 0.
            ff_dropout (float, optional): Dropout rate for feedforward layers. Default: 0.
            final_proj (bool, optional): Whether to use a final projection layer. Default: True.
            normformer (bool, optional): Whether to use post-activation normalization in feedforward layers. Default: False.
            rotary_emb (bool, optional): Whether to use rotary embeddings. Default: True.
        """
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()

        self.rel_pos_bias = RelPosBias(heads=heads)

        rotary_emb_instance = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, causal=True, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_emb=rotary_emb_instance),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, post_activation_norm=normformer)
            ]))

        self.norm = LayerNorm(dim, stable=True) if norm_out else nn.Identity()
        self.project_out = nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CausalTransformer module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the transformer layers.
        """
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device=device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)