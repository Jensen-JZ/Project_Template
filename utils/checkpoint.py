import os

import torch


class CheckpointManager:
    """
    Utility class to manage saving and loading model ckpts.

    Supports multiple registered modules (e.g., model, optimizer) and handles
    both single-GPU and multi-GPU training scenarios via `nn.DataParallel`.

    Args:
        ckpt_path_template (str): File path template for saving/loading ckpts. Must include `{}`
            placeholder for step number.
        multi_gpu (bool): Whether the modules are wrapped in `nn.DataParallel` (for multi-GPU).
        **kwargs: Modules (e.g., model, optimizer) to register for ckpt management.

    Attributes:
        ckpt_path_template (str): The formatted string for file naming (e.g., 'models/ckpt_{}.pt').
        module_dict (dict): Dictionary of registered modules to save/load.
        multi_gpu (bool): Flag to indicate if wrapped in `nn.DataParallel`.
    """

    def __init__(self, ckpt_path_template, multi_gpu=False, **kwargs):
        os.makedirs(os.path.dirname(ckpt_path_template), exist_ok=True)
        self.ckpt_path_template = ckpt_path_template
        self.module_dict = kwargs
        self.multi_gpu = multi_gpu

    def register(self, **kwargs):
        """
        Registers additional modules for checkpointing after initialization.

        Args:
            **kwargs: Keyword arguments mapping names to PyTorch modules.
        """
        self.module_dict.update(kwargs)

    def save(self, step):
        """
        Saves the state_dict of all registered modules to disk.

        Args:
            step (int): Training step or epoch number used for naming the ckpt file.
        """

        ckpt_path = self.ckpt_path_template.format(step)
        print(f"Saving checkpoint to {ckpt_path}...")
        state_dict = {}
        for name, module in self.module_dict.items():
            if self.multi_gpu:
                state_dict[name] = module.module.state_dict()
            else:
                state_dict[name] = module.state_dict()
        torch.save(state_dict, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}.")

    def load(self, step):
        """
        Loads checkpoint from a given step number.

        Args:
            step (int): Training step or epoch number used to format the ckpt file name.
        """

        ckpt_path = self.ckpt_path_template.format(step)
        print(f"Loading checkpoint from {ckpt_path}...")
        self.load_from_path(ckpt_path)
        print(f"Checkpoint loaded from {ckpt_path}.")

    def load_from_path(self, ckpt_path):
        """
        Loads ckpt from a full path.

        Args:
            ckpt_path (str): Full path to the ckpt file.
        """

        assert os.path.exists(ckpt_path), f"Checkpoint file {ckpt_path} does not exist."
        if torch.cuda.is_available():
            module_dict = torch.load(ckpt_path)
        else:
            module_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        for name, module in self.module_dict.items():
            if self.multi_gpu:
                module.module.load_state_dict(module_dict[name])
            else:
                module.load_state_dict(module_dict[name])
