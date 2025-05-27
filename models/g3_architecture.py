# models/g3_architecture.py

import torch
import torch.nn as nn
from torch import Tensor # Added for type hinting
from typing import Optional # Added for GaussianEncoding

import numpy as np
# pandas and itertools were in G3.py's imports but don't seem directly used by the classes copied.
# import pandas as pd 
# import itertools
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
# from torch.nn import TransformerEncoder, TransformerEncoderLayer # Not directly used by G3 class here
from pyproj import Proj, Transformer

# --- RFF functional replacements ---
# Functions sample_b and gaussian_encoding were copied here from models/G3/utils/rff/functional.py

def sample_b(sigma: float, size: tuple) -> Tensor:
    r"""Matrix of size :attr:`size` sampled from from :math:`\mathcal{N}(0, \sigma^2)`

    Args:
        sigma (float): standard deviation
        size (tuple): size of the matrix sampled

    See :class:`~rff.layers.GaussianEncoding` for more details
    """
    return torch.randn(size) * sigma


@torch.jit.script
def gaussian_encoding(
        v: Tensor,
        b: Tensor) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        b (Tensor): projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`

    See :class:`~rff.layers.GaussianEncoding` for more details.
    """
    # Ensure b is on the same device as v for the matrix multiplication
    b = b.to(v.device)
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
# --- End RFF functional replacements ---


class GaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features"""

    def __init__(self, sigma: Optional[float] = None,
                 input_size: Optional[float] = None,
                 encoded_size: Optional[float] = None,
                 b: Optional[Tensor] = None):
        r"""
        Args:
            sigma (Optional[float]): standard deviation
            input_size (Optional[float]): the number of input dimensions
            encoded_size (Optional[float]): the number of dimensions the `b` matrix maps to
            b (Optional[Tensor], optional): Optionally specify a :attr:`b` matrix already sampled
        Raises:
            ValueError:
                If :attr:`b` is provided and one of :attr:`sigma`, :attr:`input_size`,
                or :attr:`encoded_size` is provided. If :attr:`b` is not provided and one of
                :attr:`sigma`, :attr:`input_size`, or :attr:`encoded_size` is not provided.
        """
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required.')

            b = sample_b(sigma, (encoded_size, input_size)) # Changed from functional.sample_b
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

        Returns:
            Tensor: Tensor mapping using random fourier features of shape :math:`(N, *, 2 \cdot \text{encoded_size})`
        """
        # If v is (N, D), and b is (E, D), then v @ b.T is (N, E)
        # The original functional.gaussian_encoding might handle batching differently if * is present.
        # This dummy forward assumes v is (N, input_size) or (input_size)
        if v.ndim == 1: # Handle case where a single vector is passed
            v_batched = v.unsqueeze(0)
            result = gaussian_encoding(v_batched, self.b) # Changed from functional.gaussian_encoding
            return result.squeeze(0)
        return gaussian_encoding(v, self.b) # Changed from functional.gaussian_encoding


class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        # GaussianEncoding is now defined in this file
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x

class CustomLocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8]):
        super(CustomLocationEncoder, self).__init__()

        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

        proj_wgs84 = Proj('epsg:4326')
        proj_mercator = Proj('epsg:3857')
        self.transformer = Transformer.from_proj(proj_wgs84, proj_mercator, always_xy=True)

    def forward(self, input_coords): # Renamed 'input' to 'input_coords' to avoid python keyword clash
        # Determine the target device from model parameters if available, else use input_coords' device
        target_device = next(self.parameters()).device if len(list(self.parameters())) > 0 else input_coords.device

        lat = input_coords[:, 0].float().cpu().numpy() # Detach and move to CPU for numpy operations
        lon = input_coords[:, 1].float().cpu().numpy() # Detach and move to CPU
        
        projected_lon_lat = self.transformer.transform(lon, lat)
        location = []
        for coord in zip(*projected_lon_lat):
            location.append([coord[1],coord[0]])
        
        location = torch.Tensor(location).to(target_device) # Move to target_device
        location = location / 20037508.3427892

        location_features = torch.zeros(location.shape[0], 512).to(target_device)

        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)

        return location_features


class G3(torch.nn.Module):
    def __init__(self, device_str="cpu"): # device_str to avoid clash with self.device attribute
        super(G3, self).__init__()
        self.device = torch.device(device_str) # Store the device

        # Load CLIP model components and move them to the specified device
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_model = clip_model.vision_model.to(self.device)
        self.text_model = clip_model.text_model.to(self.device)
        self.vision_projection = clip_model.visual_projection.to(self.device)
        self.text_projection = clip_model.text_projection.to(self.device)
        
        self.vision_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_processor = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        self.logit_scale1 = nn.Parameter(torch.tensor(3.99))
        self.logit_scale2 = nn.Parameter(torch.tensor(3.99))
        self.logit_scale3 = nn.Parameter(torch.tensor(3.99))
        # Move logit_scales to device
        self.logit_scale1.data = self.logit_scale1.data.to(self.device)
        self.logit_scale2.data = self.logit_scale2.data.to(self.device)
        self.logit_scale3.data = self.logit_scale3.data.to(self.device)


        self.location_encoder = CustomLocationEncoder().to(self.device) 
        
        self.vision_projection_else_1 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)).to(self.device)
        self.text_projection_else = nn.Sequential(nn.Linear(768,768), nn.ReLU(), nn.Linear(768, 768)).to(self.device)

        self.vision_projection_else_2 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)).to(self.device)
        self.location_projection_else = nn.Sequential(nn.Linear(512,512), nn.ReLU(), nn.Linear(512, 768)).to(self.device)

        # freeze CLIP
        self.vision_model.requires_grad_(False)
        self.vision_projection.requires_grad_(False)
        self.text_model.requires_grad_(False)
        self.text_projection.requires_grad_(False)

    def forward(self, images, texts, longitude, latitude, return_loss=True):
        # Ensure inputs are on the correct device
        images = images.to(self.device)
        # texts is a dictionary from tokenizer, its tensors need to be moved
        texts = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in texts.items()}
        longitude = longitude.to(self.device)
        latitude = latitude.to(self.device)

        vision_output = self.vision_model(images)[1]
        text_output = self.text_model(**texts)[1]
        image_embeds = self.vision_projection(vision_output)
        text_embeds = self.text_projection(text_output) 
        
        this_batch_locations = torch.stack((latitude, longitude), dim=1)
        location_embeds = self.location_encoder(this_batch_locations)

        image_embeds_1 = self.vision_projection_else_1(image_embeds)
        text_embeds_1 = self.text_projection_else(text_embeds.reshape(text_embeds.shape[0], -1))
        
        image_embeds_1 = image_embeds_1 / image_embeds_1.norm(p=2, dim=-1, keepdim=True)
        text_embeds_1 = text_embeds_1 / text_embeds_1.norm(p=2, dim=-1, keepdim=True)

        logit_scale_1_exp = self.logit_scale1.exp()
        logits_per_texts_with_images = torch.matmul(text_embeds_1, image_embeds_1.t()) * logit_scale_1_exp
        logits_per_images_with_texts = logits_per_texts_with_images.t()
        
        loss_phase_1 = None
        if return_loss:
            loss_phase_1 = self.clip_loss(logits_per_texts_with_images)

        image_embeds_2 = self.vision_projection_else_2(image_embeds)
        location_embeds_2 = self.location_projection_else(location_embeds.reshape(location_embeds.shape[0], -1))

        image_embeds_2 = image_embeds_2 / image_embeds_2.norm(p=2, dim=-1, keepdim=True)
        location_embeds_2 = location_embeds_2 / location_embeds_2.norm(p=2, dim=-1, keepdim=True)

        logit_scale_2_exp = self.logit_scale2.exp()
        logits_per_locations_with_images = torch.matmul(location_embeds_2, image_embeds_2.t()) * logit_scale_2_exp
        logits_per_images_with_locations = logits_per_locations_with_images.t()
        
        loss_phase_2 = None
        if return_loss:
            loss_phase_2 = self.clip_loss(logits_per_locations_with_images)

        loss = torch.tensor(0.0, device=self.device) # Initialize loss on the correct device
        if loss_phase_1 is not None:
            loss += loss_phase_1
        if loss_phase_2 is not None:
            loss += loss_phase_2
        
        if loss_phase_1 is None and loss_phase_2 is None and return_loss:
            # This case should ideally not happen if return_loss is True
            # Or, loss should be initialized to torch.tensor(0.0, device=self.device) if appropriate
            # For now, if no specific losses are computed but return_loss is true,
            # it implies something might be wrong or unconfigured.
            # However, returning a zero tensor is safer than `0` if other ops expect a tensor.
            pass


        return {
            'logits_per_texts_with_images': logits_per_texts_with_images,
            'logits_per_images_with_texts': logits_per_images_with_texts,
            'logits_per_locations_with_images': logits_per_locations_with_images,
            'logits_per_images_with_locations': logits_per_images_with_locations,
            'logits_per_locations_with_texts': None, # Placeholder from original
            'logits_per_texts_with_locations': None, # Placeholder from original
            'loss': loss,
            'vision_output': vision_output,
            'text_output': text_output,
            'image_embeds': image_embeds,
            'text_embeds': text_embeds
        }

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
