import warnings

import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel

from utils.model import freeze_all

# Suppress huggingface cache warning logs
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


class ImageEncoder(nn.Module):
    """
    A wrapper class around HuggingFace's CLIP image encoder with an additional MLP projection head.

    This module extracts high-level image features using a frozen CLIP model and optionally projects
    them to a lower-dimensional space using a learnable MLP.

    Attributes:
        CLIP (CLIPModel): Pre-trained CLIP model for image feature extraction.
        image_processor (AutoProcessor): Processor for preprocessing images before feeding them to the CLIP model.
        mlp (nn.Sequential): A sequential MLP model for projecting the extracted features to a lower-dimensional space.
    """

    def __init__(self):
        """
        Initializes the ImageEncoder with a pre-trained CLIP image backbone and a trainable MLP projection.
        Freezes all parameters in the CLIP model.

        Returns:
            None
        """

        super().__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.CLIP.eval()  # Set the model to evaluation mode
        # HuggingFace processor for CLIP preprocessing (resizing, normalization, etc.)
        self.image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        # Learnable MLP projection head to reduce the dimensionality of the image features
        self.mlp = (
            nn.Sequential(  # Can use self.CLIP.config.projection_dim instead of 768
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 512),
            )
        )

        # Freeze pre-trained CLIP model parameters
        freeze_all(self.CLIP)

    def preprocess_image(self, image):
        """
        Preprocesses the PIL image to match the input format expected by the CLIP model.

        Args:
            image (PIL.Image): The input image to be preprocessed.

        Returns:
            Tensor: A 4D tensor of shape (1, 3, H, W) representing the preprocessed image.
        """
        # `pt` indicates that the image tensor will be in PyTorch format
        image_tensor = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return image_tensor

    def forward(self, image):
        """
        Forward pass to extract and project image features.

        Args:
            image (Tensor): A batch of preprocessed images (4D tensor of shape (B, 3, H, W)).

        Returns:
            Tensor: A 2D tensor of shape (B, 512) representing the projected image features.
        """

        image_tensor = self.CLIP.get_image_features(pixel_values=image)
        output = self.mlp(image_tensor)
        return output
