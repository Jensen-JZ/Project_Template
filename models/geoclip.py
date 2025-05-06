import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from image_encoder import ImageEncoder
from location_encoder import LocationEncoder
from PIL import Image

from utils.file import geo_file_dir, load_gps_data


class GeoCLIP(nn.Module):
    """
    GeoCLIP model that learns a shared embedding space for images and GPS coordinates.

    This model enables computing similarity between an input image and a gallery of GPS locations,
    supporting contrastive learning and top-k location retrieval tasks.

    Attributes:
        logit_scale (nn.Parameter): Logit scale parameter for scaling the cosine similarity.
        image_encoder (ImageEncoder): Image encoder module.
        location_encoder (LocationEncoder): Location encoder module.
        gps_gallery (Tensor): Preloaded GPS coordinates for the gallery.
        gps_queue (Tensor): Queue for storing GPS coordinates.
        gps_queue_ptr (Tensor): Pointer to the current position in the GPS queue.
        device (str): Device on which the model is loaded (e.g., "cpu", "cuda").
    """

    def __init__(self, from_pretrained=True, queue_size=4096):
        """
        Initializes the GeoCLIP model with optional pre-trained weights and a GPS queue for negative sampling.

        Args:
            from_pretrained (bool): If True, loads pre-trained weights for the model.
            queue_size (int): Size of the GPS queue for negative sampling. Default is 4096.
        """

        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

        self.gps_gallery = load_gps_data(
            os.path.join(geo_file_dir, "gps_gallery", "coordinates_100K.csv")
        )
        self._initialize_gps_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(geo_file_dir, "weights")
            self._load_weights()

        self.device = "cpu"

    def to(self, device):
        """
        Moves the model and submodules to the specified device.

        Args:
            device (str): The device to move the model to (e.g., "cpu", "cuda").

        Returns:
            nn.Module: The model instance with all submodules moved to the specified device.
        """

        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    @torch.no_grad()
    def _load_weights(self):
        """
        Loads pre-trained weights for the image encoder, location encoder, and logit scale.

        Returns:
            None
        """

        self.image_encoder.mlp.load_state_dict(
            torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth")
        )
        self.location_encoder.load_state_dict(
            torch.load(f"{self.weights_folder}/location_encoder_weights.pth")
        )
        self.logit_scale.data.copy_(
            torch.load(f"{self.weights_folder}/logit_scale_weights.pth")
        )

    def _initialize_gps_queue(self, queue_size):
        """
        Initializes the GPS queue and pointer for momentum-style contrastive sampling.

        Args:
            queue_size (int): Size of the GPS queue for negative sampling.

        Returns:
            None
        """

        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = F.normalize(
            self.gps_queue, dim=0
        )  # L2 normalize for each column (GPS coordinate)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """
        Dequeues the old GPS coordinates from the queue and enqueues the new ones (FIFO).

        Args:
            gps (Tensor): New GPS coordinates (gps_batch_size, 2) to be added to the queue.

        Returns:
            None
        """

        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)

        assert (
            self.queue_size % gps_batch_size == 0
        ), f"Queue size {self.queue_size} should be divisivle by batch size {gps_batch_size}"

        # Replace the old GPS coordinates from the queue (gps_str ~ gps_str + gps_batch_size) with the new ones
        self.gps_queue[:, gps_ptr : gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # increment pointer
        self.gps_queue_ptr[0] = gps_ptr  # update pointer

    def get_gps_queue(self):
        """
        Returns the current GPS queue in the shape of (queue_size, 2) instead of (2, queue_size).

        Returns:
            Tensor: The current GPS queue with shape (queue_size, 2).
        """
        return (
            self.gps_queue.t()
        )  # return the queue in the shape of (queue_size, 2) instead of (2, queue_size)

    def forward(self, image, location):
        """
        Computes the similarity between the input image embeddings and location embeddings.

        Args:
            image (Tensor): Input image tensor of shape (batch_size, 3, height, width).
            location (Tensor): Input location tensor of shape (queue_size + batch_size, 2) representing GPS coordinates.

        Returns:
            Tensor: Logits representing the similarity scores between the image and location embeddings.
        """

        # Compute image and location features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()

        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        # Compute Cosine similarity between image and location features
        logits_per_image = logit_scale * (image_features @ location_features.t())

        return logits_per_image

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """
        Predicts the top-K GPS coordinates for the input image.

        Args:
            image_path (str): Path to the input image.
            top_k (int): Number of top GPS coordinates to retrieve.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the top-K GPS coordinates and their corresponding probabilities.
        """

        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(
            dim=-1
        ).cpu()  # Compute probabilities at batch-wise

        # Get the top k GPS coordinates
        top_pred = torch.topk(probs_per_image, top_k, dim=-1)
        top_pred_gps = self.gps_gallery[
            top_pred.indices[0]
        ]  # Get the GPS coordinates of the top k predictions
        top_pred_probs = top_pred.values[0]

        return top_pred_gps, top_pred_probs
