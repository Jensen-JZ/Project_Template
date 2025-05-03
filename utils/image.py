import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import Resize


def denormalize(image_tensor):
    """
    Converts normalized image tensor from [-1, 1] to [0, 1] range.

    Args:
        image_tensor (Tensor): Normalized image tensor.

    Returns:
        Tensor: Denormalized image tensor clamped to [0, 1].
    """

    out_tensor = (image_tensor + 1) / 2
    out_tensor = torch.clamp(out_tensor, 0, 1)
    return out_tensor


def resize_image(image_tensor, size):
    """
    Resizes the input image tensor to the specified size.

    Args:
        image_tensor (Tensor): Input image tensor with shape [C, H, W].
        size (int or Tuple[int, int]): Desired output size (height, width).

    Returns:
        Tensor: Resized image tensor.
    """
    return Resize(size)(image_tensor)


def save_image(image_tensor, columns=4, filename="image.png"):
    """
    Saves a batch of images into a single grid image file.

    Args:
        image_tensor (Tensor): A batch of image tensors with shape [B, C, H, W], normalized to [-1, 1].
        columns (int): Number of images per row in the output grid.
        filename (str): Path to save the output image file.

    Returns:
        None
    """

    image_tensor = denormalize(image_tensor)
    torchvision.utils.save_image(image_tensor.cpu(), filename, nrow=columns, padding=0)


def compare_images_side_by_side(tensor_list, titles=None):
    """
    Displays multiple images side by side for visual comparison.

    Args:
        tensor_list (List[Tensor]): List of image tensors([C, H, W]) to be displayed.
        titles (List[str], optional): List of titles for each image. Defaults to None.

    Returns:
        None
    """

    n = len(tensor_list)
    plt.figure(figsize=(4 * n, 4))
    for i, tensor in enumerate(tensor_list):
        plt.subplot(1, n, i + 1)
        plt.imshow(denormalize(tensor).permute(1, 2, 0).cpu().numpy())
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis("off")
    plt.show()


def psnr_image(image_tensor1, image_tensor2, max_value=1.0):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        image_tensor1 (Tensor): Ground truth image tensor.
        image_tensor2 (Tensor): Reconstructed or generated image tensor.
        max_value (float): Maximum possible pixel value in the image. Default is 1.0.

    Returns:
        float: PSNR value in decibels (dB).
    """
    mse = torch.mean((image_tensor1 - image_tensor2) ** 2)
    if mse == 0:
        return float("inf")  # No noise, return infinity
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr.item()
