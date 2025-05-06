import torch
import torch.nn as nn
from layers import GaussianEncoding

from utils.file import geo_file_dir

# Constants for Equal Earth Projection
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336


def equal_earth_projection(coordinates):
    """
    Applies the Equal Earth projection to latitude and longitude coordinates.

    Args:
        coordinates (torch.Tensor): A tensor of shape (B, 2) containing latitude and longitude in degrees.

    Returns:
        Tensor of shape (B, 2): The projected coordinates in the range of -1 to 1.
    """
    latitude = coordinates[:, 0]
    longitude = coordinates[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(3.0) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (
        9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1
    )
    x = (
        2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)
    ) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    # Scale factor to convert the coordinates to the range of -1 to 1
    return (torch.stack((x, y), dim=1) * SF) / 180


class LocationEncoderCapsule(nn.Module):
    """
    A capsule encoder for a specific scale (sigma) using RFF encoding + MLP.

    Attributes:
        km (float): The scale parameter for the Gaussian encoding.
        capsule (nn.Sequential): A sequential model consisting of RFF encoding and MLP layers.
        head (nn.Sequential): A linear layer that reduces the output dimension to 512.
    """

    def __init__(self, sigma):
        """
        Initializes the LocationEncoderCapsule class.

        Args:
            sigma (float): The scale parameter for the Gaussian encoding.

        Returns:
            None
        """

        super().__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(
            rff_encoding,  # Outputs 512-dim features (256 * 2)
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, location):
        """
        Forward pass of the LocationEncoderCapsule.

        Args:
            location (Tensor): A tensor of shape (B, 2) containing latitude and longitude in degrees.

        Returns:
            Tensor: The encoded location features of shape (B, 512).
        """

        location_features = self.capsule(location)
        location_features = self.head(location_features)
        return location_features


class LocationEncoder(nn.Module):
    """
    Location encoder that aggregates multiple LocationEncoderCapsule instances at different scales.

    Uses the EEP, followed by multiple RFF-based capsules to produce final 512-dimensional location embeddings.

    Attributes:
        sigma_list (list): A list of scales for the Gaussian encoding.
        num_capsules (int): The number of capsules (length of sigma_list).
    """

    def __init__(self, sigma_list=[2**0, 2**4, 2**8], from_pretrained=True):
        """
        Initializes the LocationEncoder class with a list of scales.

        Args:
            sigma_list (list): A list of scales for the Gaussian encoding.
            from_pretrained (bool): If True, loads pre-trained weights.

        Returns:
            None
        """

        super().__init__()
        self.sigma_list = sigma_list
        self.num_capsules = len(sigma_list)

        for idx, sigma in enumerate(sigma_list):
            self.add_module(f"LocEncoder_{idx}", LocationEncoderCapsule(sigma))

        if from_pretrained:
            self._load_weights()

    @torch.no_grad()
    def _load_weights(self):
        """
        Loads pre-trained weights for the LocationEncoder (all capsules) from a specified directory.

        Returns:
            None
        """
        self.load_state_dict(
            torch.load(f"{geo_file_dir}/weights/location_encoder_weights.pth")
        )

    def forward(self, location):
        """
        Forward pass through the entire location encoder.

        Args:
            location (Tensor): A tensor of shape (B, 2) containing latitude and longitude in degrees.

        Returns:
            Tensor: The aggregated location features of shape (B, 512).
        """

        eep_location = equal_earth_projection(location)
        final_location_features = torch.zeros(location.shape[0], 512).to(
            location.device
        )

        for idx in range(self.num_capsules):
            final_location_features += self._modules[f"LocEncoder_{idx}"](eep_location)

        return final_location_features
