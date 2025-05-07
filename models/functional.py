import numpy as np
import torch

"""
Functional module for the Random Fourier Features (RFF) encoding.

This module contains functions for sampling from a Gaussian distribution,
creating Gaussian encodings, and generating positional encodings.
"""


def sample_b(sigma: float, size: tuple) -> torch.Tensor:
    """
    Samples a random matrix from a zero-mean Gaussian distribution with variance simga^2.

    Typically used to generate projection matrices for Gaussian random features.

    Args:
        sigma (float): Standard deviation of the Gaussian distribution.
        size (tuple): Shape of the output tensor (e.g., (encoded_dim, input_dim)).

    Returns:
        Tensor: A tensor of shape `size` sampled from a Gaussian distribution N(0, sigma^2).
    """
    return torch.randn(size) * sigma


@torch.jit.script
def gaussian_encoding(v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Applies Gaussian random feature encoding using a projection matrix.

    Projects the input vector `v` using a fixed Gaussian matrix `b`, followed by sine and cosine transformations.

    Args:
        v (Tensor): Input tensor of shape (N, *, input_dim).
        b (Tensor): Projection matrix of shape (encoded_dim, input_dim).

    Returns:
        Tensor: Encoded tensor of shape (N, *, 2 x encoded_dim).
    """
    projected_v = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(projected_v), torch.sin(projected_v)), dim=-1)


@torch.jit.script
def basic_encoding(v: torch.Tensor) -> torch.Tensor:
    """
    Applies a basic Fourier encoding without Gaussian projection.

    Computes consine and sine transformations of the input vector `v` element-wise (scaled by 2pi).

    Args:
        v (Tensor): Input tensor of shape (N, *, input_dim).

    Returns:
        Tensor: Encoded tensor of shape (N, *, 2 x input_dim).
    """
    projected_v = 2 * np.pi * v
    return torch.cat((torch.cos(projected_v), torch.sin(projected_v)), dim=-1)


@torch.jit.script
def positional_encoding(v: torch.Tensor, sigma: float, m: int) -> torch.Tensor:
    """
    Applies multi-frequency sinusoidal encoding with exponentially spaced scales.

    Encodes each input dimension using m frequency bands scaled by sigma^(j/m) for j in [0, m-1].

    Args:
        v (Tensor): Input tensor of shape (N, *, input_dim).
        sigma (float): Scaling factor for the frequency bands.
        m (int): Number of frequency bands.

    Returns:
        Tensor: Encoded tensor of shape (N, *, 2 x m x input_dim).
    """
    j = torch.arange(m, device=v.device)
    coeffs = 2 * np.pi * sigma ** (j / m)  # generate m frequency bands
    projected_v = coeffs * torch.unsqueeze(v, dim=1)  # shape (N, *, input_dim, m)
    projected_v_cat = torch.cat(
        (torch.cos(projected_v), torch.sin(projected_v)), dim=-1
    )  # shape (N, *, input_dim, 2m)
    return projected_v_cat.flatten(-2, -1)  # shape (N, *, 2 x m x input_dim)
