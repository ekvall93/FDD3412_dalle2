""" Contribution: https://github.com/lucidrains/DALLE2-pytorch """

from resize_right import resize
import torch.nn.functional as F
import torch
from torch import nn
from functools import wraps
from torch.utils.checkpoint import checkpoint

from typing import Any, Callable, Optional, Union, List, Tuple

def module_device(module: nn.Module) -> str:
    """
    Determines the device of a PyTorch module.

    Args:
        module (nn.Module): The module to check the device for.

    Returns:
        str: The name of the device ('cpu' or 'cuda') the module parameters are on.
    """
    if isinstance(module, nn.Identity):
        return 'cpu'  # It doesn't matter
    return next(module.parameters()).device

def is_list_str(x) -> bool:
    """
    Checks if an object is a list or tuple of strings.

    Args:
        x: The object to check.

    Returns:
        bool: True if `x` is a list or tuple of strings, False otherwise.
    """
    return isinstance(x, (list, tuple)) and all(isinstance(el, str) for el in x)

def pad_tuple_to_length(t: tuple, length: int, fillvalue = None) -> tuple:
    """
    Pads a tuple to a specified length.

    Args:
        t (tuple): The tuple to pad.
        length (int): The desired length of the tuple.
        fillvalue: The value to use for padding.

    Returns:
        tuple: The padded tuple.
    """
    remain_length = length - len(t)
    return t if remain_length <= 0 else (*t, *((fillvalue,) * remain_length))

def identity(t, *args, **kwargs):
    """
    Identity function that returns the input as is.

    Args:
        t: The input to be returned.
        *args: Additional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        The input `t` unchanged.
    """
    return t

def maybe(fn):
    """
    Decorator for making a function return its input if the input is None.

    Args:
        fn: The function to apply the decorator to.

    Returns:
        The decorated function.
    """
    @wraps(fn)
    def inner(x, *args, **kwargs):
        return x if x is None else fn(x, *args, **kwargs)
    return inner

def make_checkpointable(fn, **kwargs):
    """
    Makes a function or a module list checkpointable, to save memory during backpropagation.

    Args:
        fn: The function or module list to make checkpointable.
        **kwargs: Optional keyword arguments, including a condition to determine if checkpointing should be applied.

    Returns:
        The modified function or module list.
    """
    if isinstance(fn, nn.ModuleList):
        return [maybe(make_checkpointable)(el, **kwargs) for el in fn]

    condition = kwargs.pop('condition', None)

    if condition is not None and not condition(fn):
        return fn

    @wraps(fn)
    def inner(*args):
        input_needs_grad = any(isinstance(el, torch.Tensor) and el.requires_grad for el in args)

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner


def prob_mask_like(shape: Tuple[int, ...], prob: float, device: torch.device) -> torch.Tensor:
    """
    Creates a mask with a given probability for each element being True.

    Args:
        shape (Tuple[int, ...]): The shape of the output mask.
        prob (float): Probability of each element in the mask being True.
        device (torch.device): The device to place the mask tensor on.

    Returns:
        torch.Tensor: A boolean tensor where each element is True with probability 'prob'.
    """
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.rand(shape, device=device) < prob

def zero_init_(m: nn.Module):
    """
    Initializes the weights and biases of a PyTorch module to zero.

    Args:
        m (nn.Module): The PyTorch module to initialize.
    """
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)
        

def cast_tuple(val, length = None, validate = True):
    """
    Casts a value to a tuple of a specified length.

    Args:
        val (_type_): _description_
        length (_type_, optional): _description_. Defaults to None.
        validate (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if isinstance(val, list):
        val = tuple(val)

    out = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length) and validate:
        assert len(out) == length

    return out

def first(arr: Union[Tuple, list], default: Optional[any] = None) -> any:
    """
    Returns the first element of a tuple or list, or a default value if empty.

    Args:
        arr (Union[Tuple, list]): The input tuple or list.
        default (Optional[any]): The default value to return if 'arr' is empty.

    Returns:
        any: The first element of 'arr' or the default value.
    """
    return arr[0] if arr else default

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extracts elements from 'a' at indices specified in 't'.

    Args:
        a (torch.Tensor): The tensor from which to extract elements.
        t (torch.Tensor): The indices at which to extract elements.
        x_shape (Tuple[int, ...]): The shape of the input tensor 'a'.

    Returns:
        torch.Tensor: The extracted elements, reshaped to the target shape.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def exists(val) -> bool:
    """
    Checks if a value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if 'val' is not None, False otherwise.
    """
    return val is not None

def resize_image_to(
    image: torch.Tensor,
    target_image_size: int,
    clamp_range: Optional[Tuple[float, float]] = None,
    nearest: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Resizes an image tensor to a specified size.

    Args:
        image (torch.Tensor): The image tensor to resize.
        target_image_size (int): The target size to resize the image to.
        clamp_range (Optional[Tuple[float, float]]): Range to clamp the resized image.
        nearest (bool): If True, uses nearest neighbor interpolation.

    Returns:
        torch.Tensor: The resized image tensor.
    """
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    if not nearest:
        scale_factors = target_image_size / orig_image_size
        out = F.interpolate(image, scale_factors=scale_factors, **kwargs)
    else:
        out = F.interpolate(image, target_image_size, mode='nearest')

    if clamp_range:
        out = out.clamp(*clamp_range)

    return out

def l2norm(t: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a tensor using L2 norm along the last dimension.

    Args:
        t (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The L2-normalized tensor.
    """
    return F.normalize(t, dim=-1)

def is_float_dtype(dtype: torch.dtype) -> bool:
    """
    Checks if a given data type is a floating-point data type.

    Args:
        dtype (torch.dtype): The data type to check.

    Returns:
        bool: True if 'dtype' is a floating-point data type, False otherwise.
    """
    float_dtypes = (torch.float64, torch.float32, torch.float16, torch.bfloat16)
    return dtype in float_dtypes

def default(val: Any, d: Union[Callable, Any]) -> Any:
    """
    Returns the input value 'val' if it exists, otherwise returns 'd'.

    Args:
        val: The value to check.
        d (Union[Callable, Any]): The default value to return if 'val' does not exist.
                                  If 'd' is callable, it is called to generate the default value.

    Returns:
        The original value 'val' if it exists, otherwise the default value 'd'.
    """
    return val if val is not None else (d() if callable(d) else d)

def set_module_requires_grad_(module: nn.Module, requires_grad: bool):
    """
    Sets the 'requires_grad' attribute for all parameters in a PyTorch module.

    Args:
        module (nn.Module): The module to modify.
        requires_grad (bool): The value to set for 'requires_grad'.
    """
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module: nn.Module):
    """
    Freezes all layers in a PyTorch module by setting 'requires_grad' to False.

    Args:
        module (nn.Module): The module whose layers will be frozen.
    """
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module: nn.Module):
    """
    Unfreezes all layers in a PyTorch module by setting 'requires_grad' to True.

    Args:
        module (nn.Module): The module whose layers will be unfrozen.
    """
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model: nn.Module):
    """
    Sets a PyTorch model to evaluation mode and freezes all its layers.

    Args:
        model (nn.Module): The model to freeze and set to evaluation mode.
    """
    model.eval()
    freeze_all_layers_(model)

def eval_decorator(fn: Callable) -> Callable:
    """
    Decorator for a function to run a PyTorch model in evaluation mode.

    This decorator temporarily sets the model to evaluation mode while the function is executed,
    then reverts it back to its original mode.

    Args:
        fn (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.
    """
    def inner(model: nn.Module, *args, **kwargs) -> Any:
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner