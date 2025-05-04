import torch
from munch import Munch


class Fetcher:
    """
    Wraps a PyTorch DataLoader to provide iteration with automatic device transfer
    and optional random latent vector generation during training.

    Automatically resets when DataLoader is exhausted and supports flexible data structure.

    Attributes:
        dataloader (data.DataLoader): PyTorch DataLoader instance to fetch data from.
        device (torch.device): Device to which the data will be transferred (CPU or GPU).
        latent_dim (int): Dimensionality of the latent space for generative models.
        generate_noise (bool): Flag to indicate whether to generate noise for each batch.
        iter (iter): Iterator for the DataLoader.
        iter_step (int): Number of batches processed so far.
    """

    def __init__(self, dataloader, args, start_step=0):
        """
        Initializes the Fetcher to wrap a PyTorch DataLoader to suupport pre-batch noise generation and device transfer.

        This is useful for training generative models where each batch requires random latent vectors.

        Args:
            dataloader (data.DataLoader): PyTorch DataLoader instance to fetch data from.
            args (Munch): Configuration arguments, including device settings.

        Returns:
            None
        """

        self.dataloader = dataloader
        self.device = torch.device(args.device)
        self.latent_dim = args.latent_dim
        self.generate_noise = args.mode == "train"
        self.iter = iter(self.dataloader)
        self.iter_step = 0  # Track number of batches processed

        for _ in range(start_step):
            try:
                next(self.iter)
                self.iter_step += 1
            except StopIteration:
                self.iter = iter(self.dataloader)
                break

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next batch of data, including optional random noise vector `z`.

        If the DataLoader is exhausted, it resets and continues iteration.

        Returns:
            Munch: A batch containing x, y, and optionally z (random noise vector), all transferred to the specified device.
        """

        try:
            batch = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.dataloader)
            self.iter_step = 0  # Reset the step on epoch rollover
            batch = next(self.iter)

        try:
            # Convert to dict-like object if needed
            if isinstance(batch, (list, tuple)):
                inputs = Munch(x=batch[0], y=batch[1])
            elif isinstance(batch, dict) or isinstance(batch, Munch):
                inputs = Munch(batch)
            else:
                raise TypeError(
                    f"Unsupported batch type: {type(batch)}. Expected list, tuple, dict, or Munch."
                )

            if self.generate_noise:
                batch_size = inputs.x.size(0)
                inputs.z = torch.randn(batch_size, self.latent_dim)

            # Move data to the specified device
            inputs = Munch({k: v.to(self.device) for k, v in inputs.items()})
            self.iter_step += 1
            return inputs

        except Exception as e:
            print(f"[Fetcher Error] Failed to process batch: {e}")
            raise e

    def __len__(self):
        """
        Returns the number of batches in the DataLoader.

        Returns:
            int: Number of batches in the DataLoader.
        """
        return len(self.dataloader)

    def reset(self, start_step=0):
        """
        Resets the DataLoader iterator to the beginning.

        Returns:
            None
        """
        self.iter = iter(self.dataloader)
        self.iter_step = 0
        for _ in range(start_step):
            try:
                next(self.iter)
                self.iter_step += 1
            except StopIteration:
                break
