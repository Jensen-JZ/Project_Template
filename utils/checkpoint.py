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

    def register(self, *kwargs):
        
        pass


    def save(self, step):
        pass

    def load(self, step):
        pass

    def load_from_path(self, file_path):
        pass

