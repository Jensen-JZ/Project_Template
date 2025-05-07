from typing import Optional

import functional
import torch
import torch.nn as nn


class GaussianEncoding(nn.Module):
    """
    Encodes input coordinates using random Fourier features (RFF) with a fixed projection matrix.

    This layer applies a Gaussian random projection followed by sine and cosine transformations to
    approximate a shift-invariant kernel (typically RBF/Gaussian kernel). The resulting output doubles
    the encoded dimension.

    Attributes:
        b (Tensor): A fixed projection matrix of shape (encoded_size, input_size), sampled from N(0, sigma^2).
    """

    # Optional[float] is used to indicate that the parameter can be None.
    def __init__(
        self,
        sigma: Optional[float] = None,
        input_size: Optional[float] = None,
        encoded_size: Optional[float] = None,
        b: Optional[torch.Tensor] = None,
    ):
        """
        Initializes the GaussianEncoding layer.

        Args:
            sigma (Optional[float]): Standard deviation of Gaussian used to sample the projection matrix.
            input_size (Optional[float]): Input coordinate dimension.
            encoded_size (Optional[float]): Number of RFF projections (output will be 2 * encoded_size).
            b (Optional[Tensor]): Optional pre-specified projection matrix. If provided, all other parameters are ignored.

        Returns:
            None
        """

        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    "Arguments `sigma`, `input_size`, and `encoded_size` must be provided if `b` is not given."
                )
            b = functional.sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError(
                "If `b` is provided, `sigma`, `input_size`, and `encoded_size` should not be provided."
            )
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Applies the Gaussian RFF encoding to the input tensor.

        Args:
            v (Tensor): Input tensor of shape (N, *, input_size).

        Returns:
            Tensor: Encoded tensor of shape (N, *, 2 * encoded_size).
        """
        return functional.gaussian_encoding(v, self.b)


class BasicEncoding(nn.Module):
    """
    Encodes input coordinates using a basic Fourier mapping.

    This layer applies element-wise cosine and sine transformations directly to the input tensor.
    It is a simplified version of Fourier encoding without random projection.
    """

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Applies basic sine/cosine encoding to the input tensor.

        Args:
            v (Tensor): Input tensor of shape (N, *, input_size).

        Returns:
            Tensor: Encoded tensor of shape (N, *, 2 * input_size).
        """
        return functional.basic_encoding(v)


class PositionalEncoding(nn.Module):
    """
    Applies sinusoidal positional encoding with exponentially scaled frequency bands.

    This encoding mimics the Fourier features used in Transformer models and Neural Fields.
    Each input dimension is encoded across `m` frequency bands scaled by `sigma^(j/m)`.

    Attributes:
        sigma (float): Frequency scaling factor.
        m (int): Number of frequency bands per input dimension.
    """

    def __init__(self, sigma: float, m: int):
        """
        Initializes the PositionalEncoding layer.

        Args:
            sigma (float): Frequency scaling factor (typically related to the input's spatial range).
            m (int): Number of frequency bands per input dimension.

        Returns:
            None
        """

        super().__init__()
        self.sigma = sigma
        self.m = m

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Applies multi-scale positional encoding to the input tensor.

        Args:
            v (Tensor): Input tensor of shape (N, *, input_size).

        Returns:
            Tensor: Encoded tensor of shape (N, *, 2 * m * input_size).
        """
        return functional.positional_encoding(v, self.sigma, self.m)
